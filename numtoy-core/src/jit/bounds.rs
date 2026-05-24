/// NumToy custom JIT — static bound-checking and guard-zone analysis.
///
/// Runs before byte-code emission to catch structural problems:
///
/// 1. **NaN-boxing analysis** — tracks which virtual registers can carry NaN
///    (any Div, or any value derived from a Div).  Emitted as a `BoundsReport`
///    field on `NtProgram`; callers may surface it as a warning.
///
/// 2. **Overflow / width guards** — for custom-width types (the caller already
///    filters these out via `steal_sign`, but we double-check VReg indices and
///    instruction counts stay within safe compile-time limits).
///
/// 3. **Boundary Guard Zone** — a software memory fence:  every output buffer
///    is padded with `GUARD_BYTES` bytes of canary at both ends before being
///    handed to the kernel.  `verify_guard_zone` checks the canaries are intact
///    after the kernel returns, turning a silent out-of-bounds write into a
///    hard panic before it corrupts other heap memory.

use super::ir::{BinOp, Inst, NtProgram};

// ── Tunables ──────────────────────────────────────────────────────────────────

/// Bytes of canary padding on each side of every JIT output buffer.
pub const GUARD_BYTES: usize = 64;

/// Magic byte written into the guard regions.
pub const GUARD_CANARY: u8 = 0xAB;

/// Maximum sane number of instructions in one kernel.
/// Graphs that exceed this almost certainly have a cycles / infinite-expansion
/// bug and should fall back to Cranelift rather than emitting gigantic code.
pub const MAX_INSTS: usize = 4096;

/// Maximum safe number of distinct inputs per kernel.
pub const MAX_INPUTS: usize = 256;

// ── Public report type ────────────────────────────────────────────────────────

/// Human-readable diagnostic produced by [`check_program`].
#[derive(Debug, Default)]
pub struct BoundsReport {
    /// Indices of VRegs that may contain NaN at some point in the kernel.
    pub nan_vregs:         Vec<usize>,
    /// Indices of instructions that could trigger IEEE overflow (Div by ~0).
    pub overflow_sites:    Vec<usize>,
    /// `true` when the program passed all structural sanity checks.
    pub structurally_valid: bool,
    /// Human-readable description of the first violation found, if any.
    pub first_violation:   Option<String>,
}

/// Check `prog` for NaN propagation, potential overflow, and structural
/// integrity.  Returns a [`BoundsReport`]; callers decide what to do with it.
///
/// This function never returns `Err` — even a completely broken program just
/// gets `structurally_valid = false` with a description.
pub fn check_program(prog: &NtProgram) -> BoundsReport {
    let mut report = BoundsReport { structurally_valid: true, ..Default::default() };

    // ── 1. Structural sanity ─────────────────────────────────────────────────

    if prog.insts.is_empty() {
        report.structurally_valid = false;
        report.first_violation = Some("kernel has zero instructions".into());
        return report;
    }
    if prog.insts.len() > MAX_INSTS {
        report.structurally_valid = false;
        report.first_violation = Some(format!(
            "instruction count {} exceeds MAX_INSTS ({})",
            prog.insts.len(), MAX_INSTS
        ));
        return report;
    }
    if prog.num_inputs > MAX_INPUTS {
        report.structurally_valid = false;
        report.first_violation = Some(format!(
            "input count {} exceeds MAX_INPUTS ({})",
            prog.num_inputs, MAX_INPUTS
        ));
        return report;
    }
    if prog.alloc.len() != prog.num_vregs {
        report.structurally_valid = false;
        report.first_violation = Some(format!(
            "alloc table length ({}) != num_vregs ({})",
            prog.alloc.len(), prog.num_vregs
        ));
        return report;
    }

    // Verify every VReg reference in every instruction is in-range.
    for (idx, inst) in prog.insts.iter().enumerate() {
        let bad = match inst {
            Inst::Load  { dst, .. } | Inst::Const { dst, .. } => {
                *dst >= prog.num_vregs
            }
            Inst::BinOp { dst, lhs, rhs, .. } => {
                *dst >= prog.num_vregs || *lhs >= prog.num_vregs || *rhs >= prog.num_vregs
            }
            Inst::Store { src, .. } => *src >= prog.num_vregs,
        };
        if bad {
            report.structurally_valid = false;
            report.first_violation = Some(format!(
                "instruction {idx} references out-of-range VReg (num_vregs={})",
                prog.num_vregs
            ));
            return report;
        }
    }

    // ── 2. NaN-boxing analysis ───────────────────────────────────────────────
    //
    // Conservative forward dataflow: a VReg `may_nan` if:
    //   • It is produced by a Div  (denominator could be 0)
    //   • It is produced by a BinOp where at least one operand may_nan
    // Load and Const nodes are assumed NaN-free (we don't track buffer contents).

    let mut may_nan = vec![false; prog.num_vregs];
    for (idx, inst) in prog.insts.iter().enumerate() {
        match inst {
            Inst::BinOp { dst, op, lhs, rhs } => {
                let nan_from_op  = *op == BinOp::Div;   // Div may produce NaN/Inf
                let nan_from_lhs = may_nan[*lhs];
                let nan_from_rhs = may_nan[*rhs];
                if nan_from_op || nan_from_lhs || nan_from_rhs {
                    may_nan[*dst] = true;
                }
                if nan_from_op {
                    report.overflow_sites.push(idx);
                }
            }
            _ => {}
        }
    }
    report.nan_vregs = (0..prog.num_vregs).filter(|&v| may_nan[v]).collect();

    // ── 3. AdaptableFloat / custom-width checks ──────────────────────────────
    //
    // The custom JIT only handles bits=32 or bits=64 (already enforced upstream
    // by try_lower).  A defensive re-check here catches any future regression.
    if prog.bits != 32 && prog.bits != 64 {
        report.structurally_valid = false;
        report.first_violation = Some(format!(
            "unsupported element bit width {} (only 32 or 64 allowed)",
            prog.bits
        ));
    }

    report
}

// ── Boundary Guard Zone helpers ───────────────────────────────────────────────

/// Allocate `data_bytes` bytes of output buffer surrounded by canary guards.
///
/// Layout: `[GUARD_BYTES canary | data_bytes zeros | GUARD_BYTES canary]`
///
/// Returns `(full_allocation, data_slice_start_index)`.
/// The caller must pass `buf[data_start..]` as the kernel output pointer and
/// then call [`verify_guard_zone`] when the kernel returns.
pub fn alloc_guarded_buffer(data_bytes: usize) -> (Vec<u8>, usize) {
    let total = GUARD_BYTES + data_bytes + GUARD_BYTES;
    let mut buf = vec![0u8; total];
    // Stamp canaries at both ends.
    buf[..GUARD_BYTES].fill(GUARD_CANARY);
    buf[GUARD_BYTES + data_bytes..].fill(GUARD_CANARY);
    (buf, GUARD_BYTES)
}

/// Verify that both guard regions are still intact after kernel execution.
///
/// # Panics
/// Panics with a descriptive message if the canary pattern was overwritten —
/// this is the intended behaviour: a hard abort beats silent heap corruption.
pub fn verify_guard_zone(buf: &[u8], data_bytes: usize) {
    // Leading guard
    for (i, &b) in buf[..GUARD_BYTES].iter().enumerate() {
        if b != GUARD_CANARY {
            panic!(
                "JIT Boundary Guard Zone violation: leading canary byte {} \
                 was overwritten (expected 0x{:02X}, got 0x{:02X}). \
                 The kernel wrote before its output buffer.",
                i, GUARD_CANARY, b
            );
        }
    }
    // Trailing guard
    let trail_start = GUARD_BYTES + data_bytes;
    for (i, &b) in buf[trail_start..].iter().enumerate() {
        if b != GUARD_CANARY {
            panic!(
                "JIT Boundary Guard Zone violation: trailing canary byte {} \
                 was overwritten (expected 0x{:02X}, got 0x{:02X}). \
                 The kernel wrote past the end of its output buffer.",
                i, GUARD_CANARY, b
            );
        }
    }
}
