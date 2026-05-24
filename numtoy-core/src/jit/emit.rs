/// NumToy custom JIT — target-agnostic emitter trait + x86-64 implementation.
///
/// Any new ISA back-end implements `CodeEmitter` and is selected at compile
/// time via `#[cfg(target_arch)]`.  The x86-64 implementation is here;
/// AArch64 lives in `emit_aarch64.rs`.
///
/// ## `CodeEmitter` contract
/// * `emit(prog) -> Option<Vec<u8>>` — turn a validated `NtProgram` into a
///   flat byte vector of native machine code.
/// * Returns `None` if the architecture is unavailable at compile time or the
///   program is structurally empty.  The `mod.rs` entry-point then falls back
///   to Cranelift.

use super::ir::{BinOp, Inst, NtProgram};

// ─── Target-agnostic trait ────────────────────────────────────────────────────

/// Turns a validated `NtProgram` into a flat byte vector of machine code.
pub trait CodeEmitter {
    fn emit(prog: &NtProgram) -> Option<Vec<u8>>;
}

// ─── Platform dispatch ────────────────────────────────────────────────────────

/// The emitter active for this compilation target.
#[cfg(target_arch = "x86_64")]
pub type PlatformEmitter = X86_64Emitter;

#[cfg(target_arch = "aarch64")]
pub type PlatformEmitter = super::emit_aarch64::AArch64CodeEmitter;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub type PlatformEmitter = FallbackEmitter;

/// Zero-size type used on unsupported targets — always returns `None`.
pub struct FallbackEmitter;
impl CodeEmitter for FallbackEmitter {
    fn emit(_prog: &NtProgram) -> Option<Vec<u8>> { None }
}

// ─── x86-64 implementation ────────────────────────────────────────────────────
//
// Calling convention: `extern "C" fn(*const *const u8, *mut u8, usize)`
// Windows x64  : args in rcx, rdx, r8   (32-byte shadow space required)
// System V AMD64: args in rdi, rsi, rdx
//
// Fixed register assignments (callee-saved, preserved across the loop body):
//   rbx  = inputs_ptr   (*const *const u8)
//   r14  = output_ptr   (*mut u8)
//   r15  = count        (usize)
//   r10  = i            (loop index, starts at 0)
//
// Virtual registers map to xmm0..xmm13.  xmm14/xmm15 are reserved scratch.

pub struct X86_64Emitter;

impl CodeEmitter for X86_64Emitter {
    fn emit(prog: &NtProgram) -> Option<Vec<u8>> {
        emit_kernel_x86_64(prog)
    }
}

#[cfg(target_arch = "x86_64")]
pub fn emit_kernel_x86_64(prog: &NtProgram) -> Option<Vec<u8>> {
    let mut e = Assembler::new();
    e.prologue();

    let loop_top = e.label();
    e.cmp_r64_r64(R10, R15);
    let jge_fixup = e.jge_rel32_placeholder();

    for inst in &prog.insts {
        match inst {
            Inst::Load { dst, input_idx, bits } => {
                let xmm_dst = prog.alloc[*dst];
                e.load_input_ptr(RAX, RBX, *input_idx);
                if *bits == 64 {
                    e.movsd_xmm_mem_r10x8(xmm_dst, RAX);
                } else {
                    e.load_f32_to_xmm(xmm_dst, RAX);
                }
            }
            Inst::Const { dst, val, bits } => {
                let xmm_dst = prog.alloc[*dst];
                if *bits == 64 {
                    e.movq_xmm_imm64(xmm_dst, val.to_bits());
                } else {
                    e.movd_xmm_imm32(xmm_dst, ((*val as f32).to_bits()) as u64);
                    e.cvtss2sd(xmm_dst, xmm_dst);
                }
            }
            Inst::BinOp { dst, op, lhs, rhs } => {
                let xd = prog.alloc[*dst];
                let xl = prog.alloc[*lhs];
                let xr = prog.alloc[*rhs];
                if xd != xl {
                    e.movsd_xmm_xmm(xd, xl);
                }
                match op {
                    BinOp::Add => e.addsd(xd, xr),
                    BinOp::Sub => e.subsd(xd, xr),
                    BinOp::Mul => e.mulsd(xd, xr),
                    BinOp::Div => e.divsd(xd, xr),
                }
            }
            Inst::Store { src, bits } => {
                let xmm_src = prog.alloc[*src];
                if *bits == 64 {
                    e.movsd_mem_r10x8_xmm(R14, xmm_src);
                } else {
                    e.store_xmm_f32(R14, xmm_src);
                }
            }
        }
    }

    e.inc_r64(R10);
    e.jmp_rel32_back(loop_top);
    e.patch_jge(jge_fixup);
    e.epilogue();

    if e.buf.is_empty() { None } else { Some(e.buf) }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn emit_kernel_x86_64(_prog: &NtProgram) -> Option<Vec<u8>> { None }

// ─── Register constants ───────────────────────────────────────────────────────

const RAX: u8 = 0;
const RBX: u8 = 3;
const R10: u8 = 10;
const R14: u8 = 14;
const R15: u8 = 15;

// ─── Assembler ────────────────────────────────────────────────────────────────

struct Assembler {
    pub buf: Vec<u8>,
}

impl Assembler {
    fn new() -> Self { Assembler { buf: Vec::with_capacity(512) } }

    fn push_u8(&mut self, b: u8)  { self.buf.push(b); }
    fn push_u32_le(&mut self, v: u32) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn push_u64_le(&mut self, v: u64) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn label(&self) -> usize { self.buf.len() }

    fn rex(&mut self, w: bool, r_ext: bool, x_ext: bool, b_ext: bool) {
        let byte = 0x40u8
            | (if w     { 0x08 } else { 0 })
            | (if r_ext { 0x04 } else { 0 })
            | (if x_ext { 0x02 } else { 0 })
            | (if b_ext { 0x01 } else { 0 });
        self.push_u8(byte);
    }
    fn rex_w(&mut self) { self.push_u8(0x48); }
    fn modrm_rr(&self, r: u8, rm: u8) -> u8 { 0xC0 | ((r & 7) << 3) | (rm & 7) }

    // ── Prologue / Epilogue ───────────────────────────────────────────────────

    fn prologue(&mut self) {
        self.push_u8(0x55);                       // push rbp
        self.push_u8(0x48); self.push_u8(0x89); self.push_u8(0xE5); // mov rbp, rsp
        self.push_u8(0x53);                       // push rbx
        self.push_u8(0x41); self.push_u8(0x56);   // push r14
        self.push_u8(0x41); self.push_u8(0x57);   // push r15
        // sub rsp, 8  (16-byte align after 4×8-byte pushes + ret addr)
        self.push_u8(0x48); self.push_u8(0x83); self.push_u8(0xEC); self.push_u8(0x08);

        #[cfg(target_os = "windows")]
        {
            // mov rbx, rcx   — REX.W(48) 89 CB
            self.push_u8(0x48); self.push_u8(0x89); self.push_u8(0xCB);
            // mov r14, rdx   — REX.WB(49) 89 D6
            //   REX.B extends rm (r14≥8); rdx=2 needs no extension.
            self.push_u8(0x49); self.push_u8(0x89); self.push_u8(0xD6);
            // mov r15, r8    — REX.WRB(4D) 89 C7
            self.push_u8(0x4D); self.push_u8(0x89); self.push_u8(0xC7);
        }
        #[cfg(not(target_os = "windows"))]
        {
            // mov rbx, rdi   — 48 89 FB
            self.push_u8(0x48); self.push_u8(0x89); self.push_u8(0xFB);
            // mov r14, rsi   — 49 89 F6
            self.push_u8(0x49); self.push_u8(0x89); self.push_u8(0xF6);
            // mov r15, rdx   — 49 89 D7
            self.push_u8(0x49); self.push_u8(0x89); self.push_u8(0xD7);
        }
        // xor r10, r10   (i = 0)  — 4D 31 D2
        self.push_u8(0x4D); self.push_u8(0x31); self.push_u8(0xD2);
    }

    fn epilogue(&mut self) {
        self.push_u8(0x48); self.push_u8(0x83); self.push_u8(0xC4); self.push_u8(0x08); // add rsp, 8
        self.push_u8(0x41); self.push_u8(0x5F); // pop r15
        self.push_u8(0x41); self.push_u8(0x5E); // pop r14
        self.push_u8(0x5B);                     // pop rbx
        self.push_u8(0x5D);                     // pop rbp
        self.push_u8(0xC3);                     // ret
    }

    // ── Control flow ─────────────────────────────────────────────────────────

    fn cmp_r64_r64(&mut self, l: u8, r: u8) {
        self.rex(true, l >= 8, false, r >= 8);
        self.push_u8(0x3B);
        self.push_u8(self.modrm_rr(l, r));
    }
    fn jge_rel32_placeholder(&mut self) -> usize {
        self.push_u8(0x0F); self.push_u8(0x8D);
        let site = self.buf.len();
        self.push_u32_le(0);
        site
    }
    fn patch_jge(&mut self, site: usize) {
        let rel = (self.buf.len() as i32) - (site as i32 + 4);
        let bytes = rel.to_le_bytes();
        self.buf[site..site + 4].copy_from_slice(&bytes);
    }
    fn jmp_rel32_back(&mut self, loop_top: usize) {
        self.push_u8(0xE9);
        let src = (self.buf.len() + 4) as i32;
        let rel = loop_top as i32 - src;
        self.push_u32_le(rel as u32);
    }
    fn inc_r64(&mut self, r: u8) {
        self.rex(true, false, false, r >= 8);
        self.push_u8(0xFF);
        self.push_u8(0xC0 | (r & 7));
    }

    // ── Memory ───────────────────────────────────────────────────────────────

    fn load_input_ptr(&mut self, dst: u8, base: u8, input_idx: usize) {
        let disp = (input_idx * 8) as i32;
        self.rex(true, dst >= 8, false, base >= 8);
        self.push_u8(0x8B);
        self.push_u8(0x80 | ((dst & 7) << 3) | (base & 7));
        self.push_u32_le(disp as u32);
    }
    fn movsd_xmm_mem_r10x8(&mut self, xmm_dst: usize, base: u8) {
        self.push_u8(0xF2);
        self.rex(false, xmm_dst >= 8, true, base >= 8);
        self.push_u8(0x0F); self.push_u8(0x10);
        self.push_u8(((xmm_dst as u8 & 7) << 3) | 4);
        self.push_u8((3 << 6) | (2 << 3) | (base & 7));
    }
    fn movsd_mem_r10x8_xmm(&mut self, base: u8, xmm_src: usize) {
        self.push_u8(0xF2);
        self.rex(false, xmm_src >= 8, true, base >= 8);
        self.push_u8(0x0F); self.push_u8(0x11);
        self.push_u8(((xmm_src as u8 & 7) << 3) | 4);
        self.push_u8((3 << 6) | (2 << 3) | (base & 7));
    }
    fn load_f32_to_xmm(&mut self, xmm_dst: usize, base: u8) {
        // mov eax, [base + r10*4]
        self.rex(false, false, true, base >= 8);
        self.push_u8(0x8B);
        self.push_u8(4);
        self.push_u8((2 << 6) | (2 << 3) | (base & 7));
        // movd xmm_dst, eax
        self.push_u8(0x66);
        if xmm_dst >= 8 { self.push_u8(0x44); }
        self.push_u8(0x0F); self.push_u8(0x6E);
        self.push_u8(self.modrm_rr(xmm_dst as u8, RAX));
        // cvtss2sd xmm_dst, xmm_dst
        self.cvtss2sd(xmm_dst, xmm_dst);
    }
    fn store_xmm_f32(&mut self, base: u8, xmm_src: usize) {
        self.cvtsd2ss(xmm_src, xmm_src);
        // movd eax, xmm_src
        self.push_u8(0x66);
        if xmm_src >= 8 { self.push_u8(0x44); }
        self.push_u8(0x0F); self.push_u8(0x7E);
        self.push_u8(self.modrm_rr(xmm_src as u8, RAX));
        // mov [base + r10*4], eax
        self.rex(false, false, true, base >= 8);
        self.push_u8(0x89);
        self.push_u8(4);
        self.push_u8((2 << 6) | (2 << 3) | (base & 7));
    }
    fn movsd_xmm_xmm(&mut self, xmm_dst: usize, xmm_src: usize) {
        self.push_u8(0xF2);
        self.rex(false, xmm_dst >= 8, false, xmm_src >= 8);
        self.push_u8(0x0F); self.push_u8(0x10);
        self.push_u8(self.modrm_rr(xmm_dst as u8, xmm_src as u8));
    }
    fn movq_xmm_imm64(&mut self, xmm_dst: usize, bits: u64) {
        self.rex_w();
        self.push_u8(0xB8);
        self.push_u64_le(bits);
        self.push_u8(0x66);
        self.rex(true, xmm_dst >= 8, false, false);
        self.push_u8(0x0F); self.push_u8(0x6E);
        self.push_u8(self.modrm_rr(xmm_dst as u8, RAX));
    }
    fn movd_xmm_imm32(&mut self, xmm_dst: usize, bits: u64) {
        self.push_u8(0xB8);
        self.push_u32_le(bits as u32);
        self.push_u8(0x66);
        if xmm_dst >= 8 { self.push_u8(0x44); }
        self.push_u8(0x0F); self.push_u8(0x6E);
        self.push_u8(self.modrm_rr(xmm_dst as u8, RAX));
    }

    // ── SSE2 scalar double ────────────────────────────────────────────────────

    fn sse2_op_rr(&mut self, opcode: u8, dst: usize, src: usize) {
        self.push_u8(0xF2);
        if dst >= 8 || src >= 8 { self.rex(false, dst >= 8, false, src >= 8); }
        self.push_u8(0x0F); self.push_u8(opcode);
        self.push_u8(self.modrm_rr(dst as u8, src as u8));
    }
    fn addsd(&mut self, d: usize, s: usize) { self.sse2_op_rr(0x58, d, s); }
    fn subsd(&mut self, d: usize, s: usize) { self.sse2_op_rr(0x5C, d, s); }
    fn mulsd(&mut self, d: usize, s: usize) { self.sse2_op_rr(0x59, d, s); }
    fn divsd(&mut self, d: usize, s: usize) { self.sse2_op_rr(0x5E, d, s); }

    fn cvtss2sd(&mut self, dst: usize, src: usize) {
        self.push_u8(0xF3);
        if dst >= 8 || src >= 8 { self.rex(false, dst >= 8, false, src >= 8); }
        self.push_u8(0x0F); self.push_u8(0x5A);
        self.push_u8(self.modrm_rr(dst as u8, src as u8));
    }
    fn cvtsd2ss(&mut self, dst: usize, src: usize) {
        self.push_u8(0xF2);
        if dst >= 8 || src >= 8 { self.rex(false, dst >= 8, false, src >= 8); }
        self.push_u8(0x0F); self.push_u8(0x5A);
        self.push_u8(self.modrm_rr(dst as u8, src as u8));
    }
}

// ─── AArch64 wrapper (delegates to emit_aarch64) ─────────────────────────────

/// Zero-size newtype that routes `CodeEmitter::emit` to the AArch64 assembler.
#[cfg(target_arch = "aarch64")]
pub mod aarch64_adapter {
    use super::{CodeEmitter, NtProgram};
    pub struct AArch64CodeEmitter;
    impl CodeEmitter for AArch64CodeEmitter {
        fn emit(prog: &NtProgram) -> Option<Vec<u8>> {
            super::super::emit_aarch64::emit_kernel_aarch64(prog)
        }
    }
}
#[cfg(target_arch = "aarch64")]
pub use aarch64_adapter::AArch64CodeEmitter;
