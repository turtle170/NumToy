/// NumToy custom JIT — public entry point.
///
/// `try_compile` is the single function called by the execution path.
/// It chains: ArenaGraph → SSA IR → register allocation → bounds checking →
/// native code emission via the target-selected backend → executable page
/// sealing.  Returns `None` for any graph the custom JIT cannot handle;
/// callers must fall back to Cranelift.
///
/// Supported: Float(32) and Float(64) element-wise binary trees
///   (Add, Sub, Mul, Div, Constant leaves, Variable leaves).
/// Unsupported → Cranelift: DequantizeMatmul, custom bit widths, Int types.

pub mod ir;
pub mod emit;
pub mod emit_aarch64;
pub mod mem;
pub mod bounds;
pub mod backend;

use crate::cache::ExecFn;
use crate::graph::ArenaGraph;

/// Try to compile `graph` with the custom NumToy JIT.
///
/// Returns `Some(fn_ptr)` if compilation succeeds, `None` otherwise.
/// The function pointer is valid for the process lifetime (pages are leaked).
///
/// ## Guard-zone note
/// The actual output-buffer guard zone (canary allocation + post-call
/// `bounds::verify_guard_zone`) is applied at the call site in array.rs,
/// which owns the output buffer lifetime.  The `bounds` field on the
/// returned `NtProgram` (stored in the JIT cache) carries `overflow_sites`
/// and `nan_vregs` so that call sites can choose to enable extra checks
/// for kernels that may produce NaN/Inf.
pub fn try_compile(graph: &ArenaGraph) -> Option<ExecFn> {
    // Stage 1 — lower graph to SSA IR + run lifetime analysis + register allocation.
    // Structural bounds-checking (MAX_INSTS, MAX_INPUTS, vreg range) is also
    // performed inside `try_lower`; a structurally invalid program returns None.
    let prog = ir::try_lower(graph)?;

    // Stage 2 — log diagnostics from the bounds report (debug builds only).
    #[cfg(debug_assertions)]
    {
        if !prog.bounds.overflow_sites.is_empty() {
            eprintln!(
                "[numtoy-jit] backend={} overflow_sites={:?} nan_vregs={:?}",
                backend::native_backend().name(),
                prog.bounds.overflow_sites,
                prog.bounds.nan_vregs,
            );
        }
    }

    // Stage 3 — emit native machine code via the target-selected backend.
    let code = backend::native_backend().emit(&prog)?;

    // Stage 4 — seal into an executable page and return a callable pointer.
    // SAFETY: the backend guarantees `code` is valid machine code matching
    // the ExecFn ABI for this target.
    let fn_ptr = unsafe { mem::seal_exec_page(&code) }?;

    // Stage 5 — on AArch64, flush the D-cache/I-cache so the CPU sees the
    // newly written instructions.  This is a no-op on other targets.
    #[cfg(target_arch = "aarch64")]
    {
        let ptr = fn_ptr as *const u8;
        emit_aarch64::flush_icache(ptr, code.len());
    }

    Some(fn_ptr)
}
