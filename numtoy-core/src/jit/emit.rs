/// NumToy custom JIT — x86-64 machine-code emitter.
///
/// Generates a tight element-wise loop that reads from N input buffers and
/// writes to one output buffer.  All arithmetic is done in XMM registers.
///
/// Calling convention: `extern "C" fn(*const *const u8, *mut u8, usize)`
/// - Windows x64 : args in rcx, rdx, r8   (shadow space required)
/// - System V     : args in rdi, rsi, rdx
///
/// Register assignments (callee-saved, preserved across the loop body):
///   rbx  = inputs_ptr   (*const *const u8)
///   r14  = output_ptr   (*mut u8)
///   r15  = count        (usize — total element count)
///   r10  = i            (loop index, 0..count)
///
/// The loop body is generated from an `NtProgram`; each VReg maps to an
/// XMM physical register determined by the allocator.

use super::ir::{BinOp, Inst, NtProgram};

// ─── Public entry point ───────────────────────────────────────────────────────

/// Emit a complete native function for `prog` and return its bytes.
/// Returns `None` only if the program is somehow empty.
#[cfg(target_arch = "x86_64")]
pub fn emit_kernel(prog: &NtProgram) -> Option<Vec<u8>> {
    let mut e = Assembler::new();

    // ── Prologue ─────────────────────────────────────────────────────────────
    e.prologue();

    // ── Loop header: compare i < count, jump to epilogue if done ─────────────
    // r10 (i) = 0  (prologue already zeroed it)
    // loop_top label recorded here:
    let loop_top = e.label();

    // cmp r10, r15
    e.cmp_r64_r64(R10, R15);
    let jge_fixup = e.jge_rel32_placeholder();

    // ── Loop body: generate one instruction per NtProgram inst ───────────────
    for inst in &prog.insts {
        match inst {
            Inst::Load { dst, input_idx, bits } => {
                let xmm_dst = prog.alloc[*dst];
                // mov rax, [rbx + input_idx*8]   — load the slot pointer
                e.load_input_ptr(RAX, RBX, *input_idx);
                if *bits == 64 {
                    // movsd xmm_dst, [rax + r10*8]
                    e.movsd_xmm_mem_r10x8(xmm_dst, RAX);
                } else {
                    // mov eax, [rax + r10*4]; movd xmm_dst, eax; cvtss2sd xmm_dst, xmm_dst
                    e.load_f32_to_xmm(xmm_dst, RAX);
                }
            }
            Inst::Const { dst, val, bits } => {
                let xmm_dst = prog.alloc[*dst];
                // Write constant via rax: mov rax, <bits>; movd/movq xmm_dst, rax
                if *bits == 64 {
                    e.movq_xmm_imm64(xmm_dst, val.to_bits());
                } else {
                    e.movd_xmm_imm32(xmm_dst, ((*val as f32).to_bits()) as u64);
                    // cvtss2sd xmm_dst, xmm_dst
                    e.cvtss2sd(xmm_dst, xmm_dst);
                }
            }
            Inst::BinOp { dst, op, lhs, rhs } => {
                let xd = prog.alloc[*dst];
                let xl = prog.alloc[*lhs];
                let xr = prog.alloc[*rhs];
                // If dst != lhs, we need a movsd first (non-destructive form unavailable).
                // We use xmm14 as a scratch if dst collides with neither operand;
                // otherwise emit movsd dst, lhs then op dst, rhs.
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
                    // movsd [r14 + r10*8], xmm_src
                    e.movsd_mem_r10x8_xmm(R14, xmm_src);
                } else {
                    // cvtsd2ss xmm_src, xmm_src; movd eax, xmm_src; mov [r14 + r10*4], eax
                    e.store_xmm_f32(R14, xmm_src);
                }
            }
        }
    }

    // ── Loop increment and back-edge ─────────────────────────────────────────
    // inc r10
    e.inc_r64(R10);
    // jmp loop_top
    e.jmp_rel32_back(loop_top);

    // ── Patch the jge_rel32 placeholder ──────────────────────────────────────
    e.patch_jge(jge_fixup);

    // ── Epilogue ─────────────────────────────────────────────────────────────
    e.epilogue();

    if e.buf.is_empty() { None } else { Some(e.buf) }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn emit_kernel(_prog: &NtProgram) -> Option<Vec<u8>> {
    None
}

// ─── Register constants ───────────────────────────────────────────────────────

const RAX: u8 = 0;
const RBX: u8 = 3;
const R10: u8 = 10;
const R14: u8 = 14;
const R15: u8 = 15;

// ─── Assembler ───────────────────────────────────────────────────────────────

struct Assembler {
    pub buf: Vec<u8>,
}

impl Assembler {
    fn new() -> Self {
        Assembler { buf: Vec::with_capacity(512) }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn push_u8(&mut self, b: u8) { self.buf.push(b); }

    fn push_u32_le(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn push_u64_le(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn label(&self) -> usize { self.buf.len() }

    /// REX prefix: W=1 (64-bit op), R=reg extension bit, X=SIB index extension, B=rm extension.
    fn rex(&mut self, w: bool, r_ext: bool, x_ext: bool, b_ext: bool) {
        let byte = 0x40u8
            | (if w     { 0x08 } else { 0 })
            | (if r_ext { 0x04 } else { 0 })
            | (if x_ext { 0x02 } else { 0 })
            | (if b_ext { 0x01 } else { 0 });
        self.push_u8(byte);
    }

    /// REX.W only (most common case: 64-bit GPR ops involving low registers).
    fn rex_w(&mut self) { self.push_u8(0x48); }

    /// ModRM byte: mod=11 (register), reg=r, rm=m.
    fn modrm_rr(&self, r: u8, rm: u8) -> u8 { 0xC0 | ((r & 7) << 3) | (rm & 7) }

    // ── Prologue / Epilogue ───────────────────────────────────────────────────
    //
    // Frame layout (aligned to 16 bytes):
    //   push rbp; mov rbp, rsp
    //   push rbx; push r14; push r15    ← callee-saved
    //   (Windows: sub rsp, 32 for shadow space)
    //
    // Move calling-convention args into our fixed registers.

    fn prologue(&mut self) {
        // push rbp
        self.push_u8(0x55);
        // mov rbp, rsp  — REX.W 89 /5
        self.push_u8(0x48); self.push_u8(0x89); self.push_u8(0xE5);

        // push rbx
        self.push_u8(0x53);
        // push r14 — 41 56
        self.push_u8(0x41); self.push_u8(0x56);
        // push r15 — 41 57
        self.push_u8(0x41); self.push_u8(0x57);

        // Align stack: we pushed 4 × 8 = 32 bytes after rbp push (+ rbp itself = 40);
        // plus the return addr makes 48 — need 16-byte alignment so sub rsp, 8.
        // REX.W 83 EC 08
        self.push_u8(0x48); self.push_u8(0x83); self.push_u8(0xEC); self.push_u8(0x08);

        // Move args into fixed registers.
        // The args differ by platform ABI:
        //   Windows: rcx=inputs_ptr, rdx=output_ptr, r8=count
        //   SysV:    rdi=inputs_ptr, rsi=output_ptr, rdx=count
        #[cfg(target_os = "windows")]
        {
            // mov rbx, rcx   — REX.W(48) 89 ModRM(mod=11 reg=rcx=1 rm=rbx=3 → CB)
            self.push_u8(0x48); self.push_u8(0x89); self.push_u8(0xCB);
            // mov r14, rdx   — REX.WB(49) 89 ModRM(mod=11 reg=rdx=2 rm=r14&7=6 → D6)
            //   REX.B extends the rm field (r14 ≥ 8); rdx=2 needs no extension.
            //   Note: 0x4C = REX.WR would extend the *source* (reg) field — wrong!
            self.push_u8(0x49); self.push_u8(0x89); self.push_u8(0xD6);
            // mov r15, r8    — REX.WRB(4D) 89 ModRM(mod=11 reg=r8&7=0 rm=r15&7=7 → C7)
            //   Both registers ≥ 8: REX.R extends reg (r8), REX.B extends rm (r15).
            self.push_u8(0x4D); self.push_u8(0x89); self.push_u8(0xC7);
        }
        #[cfg(not(target_os = "windows"))]
        {
            // mov rbx, rdi   (48 89 FB)
            self.push_u8(0x48); self.push_u8(0x89); self.push_u8(0xFB);
            // mov r14, rsi   (49 89 F6)
            self.push_u8(0x49); self.push_u8(0x89); self.push_u8(0xF6);
            // mov r15, rdx   (49 89 D7)
            self.push_u8(0x49); self.push_u8(0x89); self.push_u8(0xD7);
        }

        // xor r10, r10   (loop counter i = 0)  — 4D 31 D2
        self.push_u8(0x4D); self.push_u8(0x31); self.push_u8(0xD2);
    }

    fn epilogue(&mut self) {
        // add rsp, 8   — 48 83 C4 08
        self.push_u8(0x48); self.push_u8(0x83); self.push_u8(0xC4); self.push_u8(0x08);
        // pop r15 — 41 5F
        self.push_u8(0x41); self.push_u8(0x5F);
        // pop r14 — 41 5E
        self.push_u8(0x41); self.push_u8(0x5E);
        // pop rbx — 5B
        self.push_u8(0x5B);
        // pop rbp — 5D
        self.push_u8(0x5D);
        // ret — C3
        self.push_u8(0xC3);
    }

    // ── Flow control ─────────────────────────────────────────────────────────

    /// Emit `cmp r10, r15` (compare loop counter against count).
    fn cmp_r64_r64(&mut self, l: u8, r: u8) {
        // REX.WRB (R=reg>7 extension, B=rm>7 extension)
        let r_ext = l >= 8;
        let b_ext = r >= 8;
        self.rex(true, r_ext, false, b_ext);
        self.push_u8(0x3B);
        self.push_u8(self.modrm_rr(l, r));
    }

    /// Emit a `jge rel32` with a placeholder offset; return patch site index.
    fn jge_rel32_placeholder(&mut self) -> usize {
        self.push_u8(0x0F); self.push_u8(0x8D); // jge rel32
        let site = self.buf.len();
        self.push_u32_le(0); // placeholder
        site
    }

    /// Patch the `jge rel32` at `site` to jump to current position.
    fn patch_jge(&mut self, site: usize) {
        let target = self.buf.len() as i32;
        let src = (site + 4) as i32; // instruction after the placeholder
        let rel = target - src;
        let bytes = rel.to_le_bytes();
        self.buf[site..site + 4].copy_from_slice(&bytes);
    }

    /// Emit `jmp rel32` back to `loop_top`.
    fn jmp_rel32_back(&mut self, loop_top: usize) {
        self.push_u8(0xE9);
        let src = (self.buf.len() + 4) as i32;
        let tgt = loop_top as i32;
        let rel = tgt - src;
        self.push_u32_le(rel as u32);
    }

    /// Record the current position as a label (returns byte offset).
    // (used by the public `label()` helper above)

    /// Emit `inc r10` — increment loop counter.
    fn inc_r64(&mut self, r: u8) {
        // REX.WB 0xFF /0 r
        self.rex(true, false, false, r >= 8);
        self.push_u8(0xFF);
        self.push_u8(0xC0 | (r & 7)); // ModRM: mod=11 reg=0 rm=r
    }

    // ── Memory / move ─────────────────────────────────────────────────────────

    /// `mov rax, [rbx + input_idx*8]` — load the i-th input buffer pointer.
    /// rbx = base (inputs array of *const u8 pointers), stride = 8 bytes.
    fn load_input_ptr(&mut self, dst: u8, base: u8, input_idx: usize) {
        let disp = (input_idx * 8) as i32;
        // REX.W 8B /r [base + disp32]
        self.rex(true, dst >= 8, false, base >= 8);
        self.push_u8(0x8B);
        // ModRM: mod=10 (disp32), reg=dst, rm=base
        self.push_u8(0x80 | ((dst & 7) << 3) | (base & 7));
        // SIB only if base == rsp/r12 (rm=4) — rbx is 3, safe; but for r12 we'd need SIB.
        // We only ever use rbx (3) as base here, so no SIB needed.
        self.push_u32_le(disp as u32);
    }

    /// `movsd xmm_dst, [rax + r10*8]`
    /// Encoding: F2 REX.XB 0F 10 /r SIB
    /// SIB: scale=3 (×8), index=r10 (10→encoded 2 + REX.X), base=rax (0)
    fn movsd_xmm_mem_r10x8(&mut self, xmm_dst: usize, base: u8) {
        self.push_u8(0xF2);
        // REX: W=0 (SSE), R=xmm_dst≥8, X=r10≥8 (yes, r10=10≥8), B=base≥8
        let r_ext = xmm_dst >= 8;
        let b_ext = base >= 8;
        self.rex(false, r_ext, true, b_ext); // X=true for r10
        self.push_u8(0x0F); self.push_u8(0x10);
        // ModRM: mod=00, reg=xmm_dst, rm=4 (SIB follows)
        self.push_u8(((xmm_dst as u8 & 7) << 3) | 4);
        // SIB: scale=3 (×8), index=r10 (encoded as 2, REX.X extends to 10), base=rax (0)
        self.push_u8((3 << 6) | (2 << 3) | (base & 7));
    }

    /// `movsd [r14 + r10*8], xmm_src`
    fn movsd_mem_r10x8_xmm(&mut self, base: u8, xmm_src: usize) {
        self.push_u8(0xF2);
        let r_ext = xmm_src >= 8;
        let b_ext = base >= 8;
        self.rex(false, r_ext, true, b_ext);
        self.push_u8(0x0F); self.push_u8(0x11);
        // ModRM: mod=00, reg=xmm_src, rm=4 (SIB)
        self.push_u8(((xmm_src as u8 & 7) << 3) | 4);
        // SIB: scale=3, index=r10 encoded as 2 (+REX.X), base=r14 encoded as 6 (+REX.B)
        self.push_u8((3 << 6) | (2 << 3) | (base & 7));
    }

    /// Load a 32-bit float from `[base + r10*4]` into xmm_dst as f64.
    /// Steps: mov eax, [base + r10*4]  →  movd xmm_dst, eax  →  cvtss2sd xmm_dst, xmm_dst
    fn load_f32_to_xmm(&mut self, xmm_dst: usize, base: u8) {
        // mov eax, [base + r10*4]
        // REX: W=0, R=0, X=1 (r10), B=base≥8
        self.rex(false, false, true, base >= 8);
        self.push_u8(0x8B);
        // ModRM: mod=00, reg=eax(0), rm=4 (SIB)
        self.push_u8(4);
        // SIB: scale=2 (×4), index=r10 encoded 2 (+REX.X=10), base=base
        self.push_u8((2 << 6) | (2 << 3) | (base & 7));

        // movd xmm_dst, eax  (66 REX 0F 6E /r)
        self.push_u8(0x66);
        if xmm_dst >= 8 { self.push_u8(0x44); } // REX.R
        self.push_u8(0x0F); self.push_u8(0x6E);
        self.push_u8(self.modrm_rr(xmm_dst as u8, RAX));

        // cvtss2sd xmm_dst, xmm_dst  (F3 REX? 0F 5A /r)
        self.cvtss2sd(xmm_dst, xmm_dst);
    }

    /// Store xmm_src (f64) as f32 at `[base + r10*4]`.
    /// Steps: cvtsd2ss xmm_src, xmm_src  →  movd eax, xmm_src  →  mov [base + r10*4], eax
    fn store_xmm_f32(&mut self, base: u8, xmm_src: usize) {
        // cvtsd2ss xmm_src, xmm_src  (F2 REX? 0F 5A /r)
        self.cvtsd2ss(xmm_src, xmm_src);

        // movd eax, xmm_src  (66 REX? 0F 7E /r)
        self.push_u8(0x66);
        if xmm_src >= 8 { self.push_u8(0x44); } // REX.R
        self.push_u8(0x0F); self.push_u8(0x7E);
        self.push_u8(self.modrm_rr(xmm_src as u8, RAX));

        // mov [base + r10*4], eax
        self.rex(false, false, true, base >= 8);
        self.push_u8(0x89);
        self.push_u8(4); // ModRM: mod=00 reg=eax rm=4 (SIB)
        self.push_u8((2 << 6) | (2 << 3) | (base & 7));
    }

    /// `movsd xmm_dst, xmm_src`
    fn movsd_xmm_xmm(&mut self, xmm_dst: usize, xmm_src: usize) {
        self.push_u8(0xF2);
        let r_ext = xmm_dst >= 8;
        let b_ext = xmm_src >= 8;
        self.rex(false, r_ext, false, b_ext);
        self.push_u8(0x0F); self.push_u8(0x10);
        self.push_u8(self.modrm_rr(xmm_dst as u8, xmm_src as u8));
    }

    /// Materialise a 64-bit float constant into xmm_dst via RAX.
    fn movq_xmm_imm64(&mut self, xmm_dst: usize, bits: u64) {
        // mov rax, imm64  (REX.W B8+rd)
        self.rex_w();
        self.push_u8(0xB8); // MOV RAX, imm64 (opcode + reg field 0)
        self.push_u64_le(bits);
        // movq xmm_dst, rax  (66 REX.W 0F 6E /r)
        self.push_u8(0x66);
        self.rex(true, xmm_dst >= 8, false, false);
        self.push_u8(0x0F); self.push_u8(0x6E);
        self.push_u8(self.modrm_rr(xmm_dst as u8, RAX));
    }

    /// Materialise a 32-bit float constant (stored as u64) into xmm_dst via RAX.
    fn movd_xmm_imm32(&mut self, xmm_dst: usize, bits: u64) {
        // mov eax, imm32  (B8 id)
        self.push_u8(0xB8);
        self.push_u32_le(bits as u32);
        // movd xmm_dst, eax  (66 REX? 0F 6E /r)
        self.push_u8(0x66);
        if xmm_dst >= 8 { self.push_u8(0x44); }
        self.push_u8(0x0F); self.push_u8(0x6E);
        self.push_u8(self.modrm_rr(xmm_dst as u8, RAX));
    }

    // ── SSE2 arithmetic (scalar double) ───────────────────────────────────────
    //
    // All have the form: F2 REX? 0F <op> /r  (register–register)

    fn sse2_op_rr(&mut self, opcode: u8, dst: usize, src: usize) {
        self.push_u8(0xF2);
        let r_ext = dst >= 8;
        let b_ext = src >= 8;
        if r_ext || b_ext { self.rex(false, r_ext, false, b_ext); }
        self.push_u8(0x0F); self.push_u8(opcode);
        self.push_u8(self.modrm_rr(dst as u8, src as u8));
    }

    fn addsd(&mut self, dst: usize, src: usize) { self.sse2_op_rr(0x58, dst, src); }
    fn subsd(&mut self, dst: usize, src: usize) { self.sse2_op_rr(0x5C, dst, src); }
    fn mulsd(&mut self, dst: usize, src: usize) { self.sse2_op_rr(0x59, dst, src); }
    fn divsd(&mut self, dst: usize, src: usize) { self.sse2_op_rr(0x5E, dst, src); }

    /// `cvtss2sd dst, src`  — F3 REX? 0F 5A /r
    fn cvtss2sd(&mut self, dst: usize, src: usize) {
        self.push_u8(0xF3);
        let r_ext = dst >= 8;
        let b_ext = src >= 8;
        if r_ext || b_ext { self.rex(false, r_ext, false, b_ext); }
        self.push_u8(0x0F); self.push_u8(0x5A);
        self.push_u8(self.modrm_rr(dst as u8, src as u8));
    }

    /// `cvtsd2ss dst, src`  — F2 REX? 0F 5A /r
    fn cvtsd2ss(&mut self, dst: usize, src: usize) {
        self.push_u8(0xF2);
        let r_ext = dst >= 8;
        let b_ext = src >= 8;
        if r_ext || b_ext { self.rex(false, r_ext, false, b_ext); }
        self.push_u8(0x0F); self.push_u8(0x5A);
        self.push_u8(self.modrm_rr(dst as u8, src as u8));
    }
}
