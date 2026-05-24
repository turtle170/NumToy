/// Executable memory page management for the NumToy custom JIT.
///
/// Allocates a read-write page, writes machine code into it, then
/// flips the protection to read-execute.  The `Allocation` is leaked
/// intentionally — the JIT cache is grow-only and pages live for the
/// process lifetime.

use crate::cache::ExecFn;

/// Write `code` into a fresh executable page and return a callable
/// `ExecFn` pointer, or `None` on any allocation / protection error.
///
/// # Safety
/// `code` must be valid x86-64 machine code matching the `ExecFn`
/// signature: `extern "C" fn(*const *const u8, *mut u8, usize)`.
pub unsafe fn seal_exec_page(code: &[u8]) -> Option<ExecFn> {
    if code.is_empty() {
        return None;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let ptr = alloc_rw_then_rx(code)?;
        // SAFETY: the emitter guarantees the byte sequence is a valid
        // function matching ExecFn's calling convention.
        Some(std::mem::transmute::<*const u8, ExecFn>(ptr))
    }

    // Non-x86-64 targets: the custom JIT is x86-64-only; callers must
    // fall back to Cranelift.
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = code;
        None
    }
}

// ── platform implementations ─────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
unsafe fn alloc_rw_then_rx(code: &[u8]) -> Option<*const u8> {
    use region::{Allocation, Protection};

    // Allocate at least 64 bytes so short kernels are cache-line aligned.
    let size = code.len().max(64);
    let mut alloc: Allocation = region::alloc(size, Protection::READ_WRITE).ok()?;

    std::ptr::copy_nonoverlapping(code.as_ptr(), alloc.as_mut_ptr::<u8>(), code.len());

    region::protect(alloc.as_ptr::<u8>(), size, Protection::READ_EXECUTE).ok()?;

    let fn_ptr = alloc.as_ptr::<u8>();

    // Leak the Allocation — its destructor would unmap the page, which
    // would invalidate the function pointer we return.
    std::mem::forget(alloc);

    Some(fn_ptr)
}
