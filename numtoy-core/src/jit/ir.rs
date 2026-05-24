/// NumToy custom JIT — SSA intermediate representation and ArenaGraph lowering.
///
/// Supports Float(32) and Float(64) binary expression trees only.  Any graph
/// that contains DequantizeMatmul, non-IEEE types, or custom bit widths returns
/// `None` from `try_lower`; callers must fall back to Cranelift.

use crate::graph::{ArenaGraph, Node, NodeId};
use crate::types::DataType;
use super::bounds::{check_program, BoundsReport};

// ─── Core SSA types ──────────────────────────────────────────────────────────

/// Virtual register index (SSA value).
pub type VReg = usize;

/// Physical XMM register (0..=7 → xmm0..xmm7, 8..=15 → xmm8..xmm15).
pub type PReg = usize;

const NUM_PHYS_REGS: usize = 14; // xmm0–xmm13; xmm14/15 reserved as scratch

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// One SSA instruction in the flat program.
#[derive(Clone, Debug)]
pub enum Inst {
    /// Load element [loop_counter] from input buffer slot `input_idx` into `dst`.
    Load { dst: VReg, input_idx: usize, bits: u32 },
    /// Broadcast scalar constant into `dst`.
    Const { dst: VReg, val: f64, bits: u32 },
    /// `dst = op(lhs, rhs)` in floating-point.
    BinOp { dst: VReg, op: BinOp, lhs: VReg, rhs: VReg },
    /// Write `src` to the output buffer at element [loop_counter].
    Store { src: VReg, bits: u32 },
}

/// Live interval [first_def .. last_use] for one virtual register.
#[derive(Clone, Copy, Debug)]
pub struct LiveInterval {
    pub vreg: VReg,
    pub start: usize, // index of defining instruction
    pub end: usize,   // index of last instruction that reads this vreg
}

/// Complete lowered program ready for native code emission.
pub struct NtProgram {
    pub insts: Vec<Inst>,
    pub num_vregs: usize,
    /// How many distinct input buffers this kernel reads.
    pub num_inputs: usize,
    /// VReg whose value is written to the output by the Store instruction.
    pub output_vreg: VReg,
    /// Element width in bits (32 or 64).
    pub bits: u32,
    /// `alloc[vreg]` → physical XMM/NEON register assigned to that VReg.
    pub alloc: Vec<PReg>,
    /// Live intervals (sorted by start) — kept for diagnostics / future passes.
    pub intervals: Vec<LiveInterval>,
    /// Static analysis report: NaN-boxing, overflow sites, guard-zone status.
    pub bounds: BoundsReport,
}

// ─── Lowering: ArenaGraph → NtProgram ────────────────────────────────────────

/// Lower an `ArenaGraph` to an `NtProgram`, or return `None` if the graph
/// contains any node that the custom JIT cannot handle.
pub fn try_lower(graph: &ArenaGraph) -> Option<NtProgram> {
    // Quick check on root type — only standard IEEE floats supported here.
    let bits = match graph.data_type(graph.root) {
        DataType::Float(32) => 32u32,
        DataType::Float(64) => 64u32,
        _ => return None, // custom widths, ints, matmul → Cranelift
    };

    // Guard: every Variable node must have steal_sign=false.
    //
    // When steal_sign=true NumToy packs floats in a custom format where the
    // IEEE 754 sign bit is reallocated to mantissa — the raw bytes are NOT
    // valid IEEE 754 and cannot be read/written directly.  When steal_sign=false
    // the encoding is bit-identical to standard IEEE 754 f32/f64, which is what
    // our direct movsd/movss loads/stores produce.
    for node in &graph.nodes {
        if let Node::Variable { steal_sign, .. } = node {
            if *steal_sign {
                return None; // non-IEEE encoding → Cranelift
            }
        }
    }

    let mut ctx = LowerCtx {
        insts: Vec::new(),
        vreg_ctr: 0,
        // Maps graph NodeId → VReg holding that node's result.
        node_vreg: vec![usize::MAX; graph.nodes.len()],
        input_map: Vec::new(), // ordered unique variable ids
    };

    let out_vreg = ctx.lower_node(graph, graph.root, bits)?;

    let num_inputs = ctx.input_map.len();
    let num_vregs = ctx.vreg_ctr;

    // Emit a store of the output VReg into the output buffer.
    ctx.insts.push(Inst::Store { src: out_vreg, bits });

    let insts = ctx.insts;

    // ── Lifetime analysis ────────────────────────────────────────────────────
    let intervals = compute_intervals(&insts, num_vregs);

    // ── Register allocation (linear scan) ───────────────────────────────────
    let alloc = linear_scan_alloc(&intervals, num_vregs)?;

    let mut prog = NtProgram {
        insts,
        num_vregs,
        num_inputs,
        output_vreg: out_vreg,
        bits,
        alloc,
        intervals,
        bounds: BoundsReport::default(),
    };

    // ── Bound-checking assertions ────────────────────────────────────────────
    // Run the static analyser BEFORE handing the program to the emitter.
    // A structurally invalid program is rejected here rather than producing
    // corrupt machine code.
    let report = check_program(&prog);
    if !report.structurally_valid {
        // Structural violation — do not emit. Fall back to Cranelift.
        return None;
    }
    prog.bounds = report;

    Some(prog)
}

// ─── Lowering context ────────────────────────────────────────────────────────

struct LowerCtx {
    insts: Vec<Inst>,
    vreg_ctr: usize,
    node_vreg: Vec<VReg>,
    /// Ordered list of variable node ids seen so far — index = input slot.
    input_map: Vec<usize>,
}

impl LowerCtx {
    fn fresh_vreg(&mut self) -> VReg {
        let v = self.vreg_ctr;
        self.vreg_ctr += 1;
        v
    }

    /// Recursively lower `node_id`.  Returns the VReg holding that node's value,
    /// or `None` if the node type is unsupported.
    fn lower_node(&mut self, graph: &ArenaGraph, node_id: NodeId, bits: u32) -> Option<VReg> {
        // Check memoization: if already lowered, reuse the VReg.
        if self.node_vreg[node_id] != usize::MAX {
            return Some(self.node_vreg[node_id]);
        }

        let vreg = match &graph.nodes[node_id] {
            Node::Variable { id, dtype, .. } => {
                // Only standard IEEE floats pass through.
                match dtype {
                    DataType::Float(32) | DataType::Float(64) => {}
                    _ => return None,
                }
                // Assign an input slot for this variable (dedup by id).
                let var_id = *id;
                let input_idx = if let Some(pos) = self.input_map.iter().position(|&x| x == var_id) {
                    pos
                } else {
                    let pos = self.input_map.len();
                    self.input_map.push(var_id);
                    pos
                };
                let dst = self.fresh_vreg();
                self.insts.push(Inst::Load { dst, input_idx, bits });
                dst
            }

            Node::Constant { val } => {
                let v = val.to_double();
                let dst = self.fresh_vreg();
                self.insts.push(Inst::Const { dst, val: v, bits });
                dst
            }

            Node::Add(l, r) => self.lower_binop(graph, *l, *r, BinOp::Add, bits)?,
            Node::Sub(l, r) => self.lower_binop(graph, *l, *r, BinOp::Sub, bits)?,
            Node::Mul(l, r) => self.lower_binop(graph, *l, *r, BinOp::Mul, bits)?,
            Node::Div(l, r) => self.lower_binop(graph, *l, *r, BinOp::Div, bits)?,

            // DequantizeMatmul is a specialised call — fall back to Cranelift.
            Node::DequantizeMatmul(..) => return None,
        };

        self.node_vreg[node_id] = vreg;
        Some(vreg)
    }

    fn lower_binop(
        &mut self,
        graph: &ArenaGraph,
        l: NodeId,
        r: NodeId,
        op: BinOp,
        bits: u32,
    ) -> Option<VReg> {
        let lhs = self.lower_node(graph, l, bits)?;
        let rhs = self.lower_node(graph, r, bits)?;
        let dst = self.fresh_vreg();
        self.insts.push(Inst::BinOp { dst, op, lhs, rhs });
        Some(dst)
    }
}

// ─── Lifetime analysis ───────────────────────────────────────────────────────

fn compute_intervals(insts: &[Inst], num_vregs: usize) -> Vec<LiveInterval> {
    let mut first_def = vec![usize::MAX; num_vregs];
    let mut last_use  = vec![0usize;        num_vregs];

    for (idx, inst) in insts.iter().enumerate() {
        match inst {
            Inst::Load  { dst, .. } | Inst::Const { dst, .. } => {
                if first_def[*dst] == usize::MAX { first_def[*dst] = idx; }
                last_use[*dst] = last_use[*dst].max(idx);
            }
            Inst::BinOp { dst, lhs, rhs, .. } => {
                if first_def[*dst] == usize::MAX { first_def[*dst] = idx; }
                last_use[*dst] = last_use[*dst].max(idx);
                last_use[*lhs] = last_use[*lhs].max(idx);
                last_use[*rhs] = last_use[*rhs].max(idx);
            }
            Inst::Store { src, .. } => {
                last_use[*src] = last_use[*src].max(idx);
            }
        }
    }

    let mut intervals: Vec<LiveInterval> = (0..num_vregs)
        .filter(|&v| first_def[v] != usize::MAX)
        .map(|v| LiveInterval { vreg: v, start: first_def[v], end: last_use[v] })
        .collect();

    intervals.sort_by_key(|iv| iv.start);
    intervals
}

// ─── Linear-scan register allocator ─────────────────────────────────────────

/// Returns `alloc[vreg] = physical_xmm_reg`, or `None` if more than
/// `NUM_PHYS_REGS` registers are simultaneously live (extremely rare for
/// element-wise kernels — fall back to Cranelift in that case).
fn linear_scan_alloc(intervals: &[LiveInterval], num_vregs: usize) -> Option<Vec<PReg>> {
    let mut alloc = vec![0usize; num_vregs];
    // free[r] = true when physical register r is available.
    let mut free = [true; NUM_PHYS_REGS];
    // active[(end, vreg)] — sorted set of live intervals currently assigned.
    let mut active: Vec<(usize, VReg, PReg)> = Vec::new(); // (end, vreg, preg)

    for iv in intervals {
        // Expire intervals whose end is before this interval's start.
        active.retain(|&(end, _, preg)| {
            if end < iv.start {
                free[preg] = true;
                false
            } else {
                true
            }
        });

        // Allocate a free physical register.
        let preg = free.iter().position(|&f| f)?; // None → spill needed → fall back
        free[preg] = false;
        alloc[iv.vreg] = preg;
        active.push((iv.end, iv.vreg, preg));
    }

    Some(alloc)
}
