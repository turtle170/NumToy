/// NumToy custom JIT — target-agnostic backend trait.
///
/// Each architecture provides a `KernelBackend` implementation that turns a
/// validated `NtProgram` into a flat byte vector of native machine code.
/// `native_backend()` returns the right one for the current build target.

/// Target-agnostic backend trait — each architecture implements `emit`.
pub trait KernelBackend: Send + Sync {
    /// Emit native machine code for `prog`.
    /// Returns `None` if emission fails or this backend is inactive on the
    /// current build target.
    fn emit(&self, prog: &super::ir::NtProgram) -> Option<Vec<u8>>;

    /// Human-readable name, e.g. "x86_64" or "aarch64".
    fn name(&self) -> &'static str;
}

/// x86-64 backend — delegates to `emit::emit_kernel_x86_64`.
pub struct X86_64Backend;
impl KernelBackend for X86_64Backend {
    fn emit(&self, prog: &super::ir::NtProgram) -> Option<Vec<u8>> {
        super::emit::emit_kernel_x86_64(prog)
    }
    fn name(&self) -> &'static str { "x86_64" }
}

/// AArch64 backend — delegates to `emit_aarch64::emit_kernel_aarch64`.
pub struct AArch64Backend;
impl KernelBackend for AArch64Backend {
    fn emit(&self, prog: &super::ir::NtProgram) -> Option<Vec<u8>> {
        super::emit_aarch64::emit_kernel_aarch64(prog)
    }
    fn name(&self) -> &'static str { "aarch64" }
}

/// Returns the native backend for the current build target.
pub fn native_backend() -> &'static dyn KernelBackend {
    #[cfg(target_arch = "x86_64")]
    { static B: X86_64Backend = X86_64Backend; &B }
    #[cfg(target_arch = "aarch64")]
    { static B: AArch64Backend = AArch64Backend; &B }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { panic!("NumToy custom JIT: no backend for this architecture") }
}
