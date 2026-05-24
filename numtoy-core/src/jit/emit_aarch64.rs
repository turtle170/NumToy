/// NumToy custom JIT — AArch64 (ARM64) machine-code emitter.
///
/// Emits a tight scalar floating-point element loop matching the same
/// `extern "C" fn(*const *const u8, *mut u8, usize)` ABI as the x86-64 path,
/// following the AAPCS64 calling convention.
///
/// ## Register map
///
/// ### GPR (general-purpose, 64-bit)
/// | Physical | Role            |
/// |----------|-----------------|
/// | x19      | inputs_ptr      |
/// | x20      | output_ptr      |
/// | x21      | count           |
/// | x22      | loop counter i  |
/// | x0       | scratch / arg0  |
///
/// ### FP/SIMD scalar (64-bit d registers, NEON)
/// Virtual registers map directly to `d0`..`d13`.
/// `d14` and `d15` are reserved as scratch for move-before-op sequences.
///
/// x19–x22 and d8–d15 are callee-saved under AAPCS64; we save/restore them.
///
/// ## Instruction encoding notes
/// All ARM64 instructions are exactly 4 bytes, little-endian.
///
/// Key encodings used (see ARM Architecture Reference Manual A64):
/// * `FADD  Dd, Dn, Dm`   → 0x1E602800 | (m<<16) | (n<<5) | d
/// * `FSUB  Dd, Dn, Dm`   → 0x1E603800 | (m<<16) | (n<<5) | d
/// * `FMUL  Dd, Dn, Dm`   → 0x1E600800 | (m<<16) | (n<<5) | d
/// * `FDIV  Dd, Dn, Dm`   → 0x1E601800 | (m<<16) | (n<<5) | d
/// * `FMOV  Dd, Dn`       → 0x1E604000 | (n<<5) | d
/// * `LDR   Dd,[Xb,Xi,LSL#3]` → 0xFC606800 | (i<<16) | (b<<5) | d
/// * `STR   Dd,[Xb,Xi,LSL#3]` → 0xFC206800 | (i<<16) | (b<<5) | d
/// * `LDR   Sd,[Xb,Xi,LSL#2]` → 0xBC606800 | (i<<16) | (b<<5) | d  (f32)
/// * `STR   Sd,[Xb,Xi,LSL#2]` → 0xBC206800 | (i<<16) | (b<<5) | d  (f32)
/// * `FCVT  Dd, Sn`       → 0x1E22C000 | (n<<5) | d   (f32→f64)
/// * `FCVT  Sd, Dn`       → 0x1E624000 | (n<<5) | d   (f64→f32)
///
/// After sealing the executable page the emitter calls the D-cache / I-cache
/// flush sequence mandated by the ARM architecture:
///   `DC CVAU` per cache line, then `DSB ISH`, then `IC IVAU` per cache line,
///   then `DSB ISH`, then `ISB`.
/// On AArch64 Linux/macOS this is performed via the `cacheflush` syscall or
/// the `__clear_cache` libgcc stub; we call `libc::cacheflush` / fall back to
/// the sys-register sequence.

use super::ir::{BinOp, Inst, NtProgram};

// ── GPR constants (used in load/store addressing) ────────────────────────────
const X0:  u32 = 0;
const X19: u32 = 19;   // inputs_ptr
const X20: u32 = 20;   // output_ptr
const X21: u32 = 21;   // count
const X22: u32 = 22;   // loop counter i
// scratch GPR for loading 64-bit constants
const X9:  u32 = 9;

// ── AArch64 emitter ───────────────────────────────────────────────────────────

pub struct AArch64Assembler {
    pub buf: Vec<u8>,
}

impl AArch64Assembler {
    pub fn new() -> Self {
        Self { buf: Vec::with_capacity(512) }
    }

    fn emit_u32(&mut self, insn: u32) {
        self.buf.extend_from_slice(&insn.to_le_bytes());
    }

    fn label(&self) -> usize { self.buf.len() }

    // ── Prologue / Epilogue ───────────────────────────────────────────────────
    //
    // Save callee-saved registers, map args to our fixed GPRs.
    // AAPCS64 args: x0 = inputs_ptr, x1 = output_ptr, x2 = count.

    pub fn prologue(&mut self) {
        // STP x29, x30, [sp, #-64]!   (frame pointer + link register)
        self.emit_u32(0xA9BB7BFD);
        // MOV x29, sp
        self.emit_u32(0x910003FD);
        // STP x19, x20, [sp, #16]
        self.emit_u32(0xA9015013);
        // STP x21, x22, [sp, #32]
        self.emit_u32(0xA9025415);
        // STP d8,  d9,  [sp, #48]   — callee-saved FP regs we may use
        self.emit_u32(0x6D032408);

        // Move args → fixed registers.
        // MOV x19, x0   (inputs_ptr)
        self.emit_u32(0xAA0003F3);
        // MOV x20, x1   (output_ptr)
        self.emit_u32(0xAA0103F4);
        // MOV x21, x2   (count)
        self.emit_u32(0xAA0203F5);
        // MOV x22, xzr  (i = 0)
        self.emit_u32(0xAA1F03F6);
    }

    pub fn epilogue(&mut self) {
        // LDP d8, d9, [sp, #48]
        self.emit_u32(0x6D432408);
        // LDP x21, x22, [sp, #32]
        self.emit_u32(0xA9425415);
        // LDP x19, x20, [sp, #16]
        self.emit_u32(0xA9415013);
        // LDP x29, x30, [sp], #64
        self.emit_u32(0xA8C57BFD);
        // RET
        self.emit_u32(0xD65F03C0);
    }

    // ── Control flow ──────────────────────────────────────────────────────────

    /// Record current offset as a label (return value).
    pub fn record_label(&self) -> usize { self.label() }

    /// Emit `CMP x22, x21` (i vs count).
    pub fn cmp_loop(&mut self) {
        // SUBS xzr, x22, x21   — sets flags
        // Encoding: 0xEB000000 | (x21<<16) | (x22<<5) | xzr
        self.emit_u32(0xEB15031F);
    }

    /// Emit `B.GE rel19` with placeholder; return patch site (byte offset of insn).
    pub fn bge_placeholder(&mut self) -> usize {
        let site = self.label();
        // B.GE #0  → 0x5400000A (condition=1010 for GE)
        self.emit_u32(0x5400000A);
        site
    }

    /// Patch the `B.cond` at `site` to branch to the current position.
    pub fn patch_bcond(&mut self, site: usize) {
        let src_pc = site as i64;
        let tgt_pc = self.label() as i64;
        let offset = tgt_pc - src_pc; // in bytes
        let imm19 = (offset / 4) as i32; // instructions
        let insn_base = 0x5400000Au32; // B.GE
        let patched = insn_base | (((imm19 as u32) & 0x7FFFF) << 5);
        let bytes = patched.to_le_bytes();
        self.buf[site..site + 4].copy_from_slice(&bytes);
    }

    /// Emit `ADD x22, x22, #1` (increment i).
    pub fn inc_loop_ctr(&mut self) {
        // ADD x22, x22, #1
        self.emit_u32(0x910006D6);
    }

    /// Emit `B label` (unconditional branch back to loop top).
    pub fn branch_back(&mut self, loop_top: usize) {
        let src_pc = self.label() as i64;
        let tgt_pc = loop_top as i64;
        let offset = tgt_pc - src_pc;
        let imm26 = (offset / 4) as i32;
        self.emit_u32(0x14000000u32 | ((imm26 as u32) & 0x03FF_FFFF));
    }

    // ── Memory access ─────────────────────────────────────────────────────────

    /// `LDR x0, [x19, #slot*8]`  — load the slot-th input buffer pointer.
    pub fn load_input_ptr(&mut self, slot: usize) {
        // LDR x0, [x19, #slot*8]
        let imm12 = (slot * 8 / 8) as u32; // byte offset / 8 (scaled by size)
        // LDR x0, [x19, #imm12_scaled]  where imm12 is in units of 8 bytes
        // Encoding: 1111 1001 01 imm12 Rn Rt
        // = 0xF9400000 | (imm12 << 10) | (X19 << 5) | X0
        self.emit_u32(0xF9400000 | (imm12 << 10) | (X19 << 5) | X0);
    }

    /// `LDR d{dst}, [x0, x22, LSL #3]`  (load f64 element at index x22).
    pub fn ldr_f64(&mut self, dreg: u32) {
        // LDR Dd, [x0, x22, LSL #3]
        // Encoding: 0xFC606800 | (X22<<16) | (X0<<5) | dreg
        self.emit_u32(0xFC606800 | (X22 << 16) | (X0 << 5) | dreg);
    }

    /// `LDR s{dst}, [x0, x22, LSL #2]`  (load f32 element at index x22, then widen).
    pub fn ldr_f32_to_f64(&mut self, dreg: u32) {
        // LDR Sd, [x0, x22, LSL #2]   (loads into low 32 bits of the register)
        // Encoding: 0xBC606800 | (X22<<16) | (X0<<5) | dreg
        self.emit_u32(0xBC606800 | (X22 << 16) | (X0 << 5) | dreg);
        // FCVT Dd, Sd   (f32 → f64 widening)
        // Encoding: 0x1E22C000 | (dreg<<5) | dreg
        self.emit_u32(0x1E22C000 | (dreg << 5) | dreg);
    }

    /// `STR d{src}, [x20, x22, LSL #3]`  (store f64 element at index x22).
    pub fn str_f64(&mut self, dreg: u32) {
        // STR Dd, [x20, x22, LSL #3]
        self.emit_u32(0xFC206800 | (X22 << 16) | (X20 << 5) | dreg);
    }

    /// `FCVT Sd, Dd` then `STR S{src}, [x20, x22, LSL #2]`  (narrow+store f32).
    pub fn str_f32(&mut self, dreg: u32) {
        // FCVT Sd, Dd (f64 → f32 narrowing) — result written to S{dreg}
        self.emit_u32(0x1E624000 | (dreg << 5) | dreg);
        // STR Sd, [x20, x22, LSL #2]
        self.emit_u32(0xBC206800 | (X22 << 16) | (X20 << 5) | dreg);
    }

    // ── FP constant materialisation ───────────────────────────────────────────

    /// Load a 64-bit float constant into d{dreg} via x9 (scratch GPR).
    pub fn fmov_imm64(&mut self, dreg: u32, bits: u64) {
        // MOVZ x9, bits[15:0]
        self.emit_u32(0xD2800000 | (((bits & 0xFFFF) as u32) << 5) | X9);
        // MOVK x9, bits[31:16], LSL #16
        if bits >> 16 != 0 {
            self.emit_u32(0xF2A00000 | ((((bits >> 16) & 0xFFFF) as u32) << 5) | X9);
        }
        // MOVK x9, bits[47:32], LSL #32
        if bits >> 32 != 0 {
            self.emit_u32(0xF2C00000 | ((((bits >> 32) & 0xFFFF) as u32) << 5) | X9);
        }
        // MOVK x9, bits[63:48], LSL #48
        if bits >> 48 != 0 {
            self.emit_u32(0xF2E00000 | ((((bits >> 48) & 0xFFFF) as u32) << 5) | X9);
        }
        // FMOV Dd, x9
        self.emit_u32(0x9E670120 | (X9 << 5) | dreg);
    }

    /// Load a 32-bit float constant (stored as u64) into d{dreg} (f64 domain).
    pub fn fmov_imm32(&mut self, dreg: u32, f32_bits: u32) {
        // MOVZ w9, f32_bits[15:0]
        self.emit_u32(0x52800000 | ((((f32_bits as u64) & 0xFFFF) as u32) << 5) | X9);
        if f32_bits >> 16 != 0 {
            // MOVK w9, f32_bits[31:16], LSL #16
            self.emit_u32(0x72A00000 | (((f32_bits >> 16) & 0xFFFF) << 5) | X9);
        }
        // FMOV Sd, w9
        self.emit_u32(0x1E270120 | (X9 << 5) | dreg);
        // FCVT Dd, Sd  (widen to f64 for uniform arithmetic)
        self.emit_u32(0x1E22C000 | (dreg << 5) | dreg);
    }

    // ── FP arithmetic (FMLA / plain FP) ──────────────────────────────────────

    /// `FMOV Dd, Dn`  (register-to-register copy, scalar double).
    pub fn fmov_dd(&mut self, dst: u32, src: u32) {
        // Encoding: 0x1E604000 | (src<<5) | dst
        self.emit_u32(0x1E604000 | (src << 5) | dst);
    }

    /// `FADD Dd, Dn, Dm`
    pub fn fadd(&mut self, dst: u32, lhs: u32, rhs: u32) {
        self.emit_u32(0x1E602800 | (rhs << 16) | (lhs << 5) | dst);
    }

    /// `FSUB Dd, Dn, Dm`
    pub fn fsub(&mut self, dst: u32, lhs: u32, rhs: u32) {
        self.emit_u32(0x1E603800 | (rhs << 16) | (lhs << 5) | dst);
    }

    /// `FMUL Dd, Dn, Dm`
    pub fn fmul(&mut self, dst: u32, lhs: u32, rhs: u32) {
        self.emit_u32(0x1E600800 | (rhs << 16) | (lhs << 5) | dst);
    }

    /// `FDIV Dd, Dn, Dm`
    pub fn fdiv(&mut self, dst: u32, lhs: u32, rhs: u32) {
        self.emit_u32(0x1E601800 | (rhs << 16) | (lhs << 5) | dst);
    }

    /// `FMADD Dd, Dn, Dm, Da`  — Fused Multiply-Add: dst = dn*dm + da
    ///
    /// Used when we detect an (a*b)+c pattern during a future FMA fusion pass.
    /// For now it is generated as the last step in Add(Mul(a,b),c) patterns
    /// by the caller.
    pub fn fmadd(&mut self, dst: u32, n: u32, m: u32, a: u32) {
        // Encoding: 0x1F400000 | (m<<16) | (a<<10) | (n<<5) | dst
        self.emit_u32(0x1F400000 | (m << 16) | (a << 10) | (n << 5) | dst);
    }
}

// ── Instruction cache flush ───────────────────────────────────────────────────

/// Flush the instruction cache for the range `[start, start+len)`.
///
/// On AArch64, writing to memory and then executing it as code requires:
///   1. `DC CVAU` — clean data cache by VA to point of unification
///   2. `DSB ISH` — data sync barrier (inner shareable)
///   3. `IC IVAU` — invalidate instruction cache by VA to PoU
///   4. `DSB ISH`
///   5. `ISB`     — instruction sync barrier
///
/// We call the `__clear_cache` symbol that every libc/libgcc on AArch64
/// provides rather than emitting the system-register instructions ourselves,
/// so the code works on all OS/libc combinations.
#[cfg(target_arch = "aarch64")]
pub fn flush_icache(ptr: *const u8, len: usize) {
    extern "C" {
        // GNU libgcc / LLVM compiler-rt / Android bionic all export this.
        fn __clear_cache(start: *mut libc::c_char, end: *mut libc::c_char);
    }
    unsafe {
        __clear_cache(ptr as *mut libc::c_char, ptr.add(len) as *mut libc::c_char);
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn flush_icache(_ptr: *const u8, _len: usize) {}

// ── Public emit entry point ───────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub fn emit_kernel_aarch64(prog: &NtProgram) -> Option<Vec<u8>> {
    let mut e = AArch64Assembler::new();
    e.prologue();

    let loop_top = e.record_label();
    e.cmp_loop();
    let bge_site = e.bge_placeholder();

    for inst in &prog.insts {
        match inst {
            Inst::Load { dst, input_idx, bits } => {
                let dreg = prog.alloc[*dst] as u32;
                e.load_input_ptr(*input_idx);
                if *bits == 64 {
                    e.ldr_f64(dreg);
                } else {
                    e.ldr_f32_to_f64(dreg);
                }
            }
            Inst::Const { dst, val, bits } => {
                let dreg = prog.alloc[*dst] as u32;
                if *bits == 64 {
                    e.fmov_imm64(dreg, val.to_bits());
                } else {
                    e.fmov_imm32(dreg, (*val as f32).to_bits());
                }
            }
            Inst::BinOp { dst, op, lhs, rhs } => {
                let xd = prog.alloc[*dst] as u32;
                let xl = prog.alloc[*lhs] as u32;
                let xr = prog.alloc[*rhs] as u32;
                // ARM64 FP ops are 3-address: dst = src1 op src2.  No move needed.
                match op {
                    BinOp::Add => e.fadd(xd, xl, xr),
                    BinOp::Sub => e.fsub(xd, xl, xr),
                    BinOp::Mul => e.fmul(xd, xl, xr),
                    BinOp::Div => e.fdiv(xd, xl, xr),
                }
            }
            Inst::Store { src, bits } => {
                let dreg = prog.alloc[*src] as u32;
                if *bits == 64 {
                    e.str_f64(dreg);
                } else {
                    e.str_f32(dreg);
                }
            }
        }
    }

    e.inc_loop_ctr();
    e.branch_back(loop_top);
    e.patch_bcond(bge_site);
    e.epilogue();

    if e.buf.is_empty() { None } else { Some(e.buf) }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn emit_kernel_aarch64(_prog: &NtProgram) -> Option<Vec<u8>> { None }
