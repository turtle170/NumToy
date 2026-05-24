/// NumToy custom JIT — public entry point.
///
/// `try_compile` is the single function called by the execution path.
/// It chains: ArenaGraph → SSA IR → register allocation → x86-64 emission →
/// executable page sealing.  Returns `None` for any graph the custom JIT
/// cannot handle; callers must fall back to Cranelift.
///
/// Supported: Float(32) and Float(64) element-wise binary trees
///   (Add, Sub, Mul, Div, Constant leaves, Variable leaves).
/// Unsupported → Cranelift: DequantizeMatmul, custom bit widths, Int types.

pub mod ir;
pub mod emit;
pub mod mem;

use crate::cache::ExecFn;
use crate::graph::ArenaGraph;

/// Try to compile `graph` with the custom NumToy JIT.
///
/// Returns `Some(fn_ptr)` if compilation succeeds, `None` otherwise.
/// The function pointer is valid for the process lifetime (pages are leaked).
pub fn try_compile(graph: &ArenaGraph) -> Option<ExecFn> {
    // Stage 1 — lower graph to SSA IR + run lifetime analysis + register allocation.
    let prog = ir::try_lower(graph)?;

    // Stage 2 — emit x86-64 machine code.
    let code = emit::emit_kernel(&prog)?;

    // Stage 3 — seal into an executable page and return a callable pointer.
    // SAFETY: `emit_kernel` produces valid x86-64 matching the ExecFn ABI.
    unsafe { mem::seal_exec_page(&code) }
}
