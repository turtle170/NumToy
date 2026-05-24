use crate::ir::Expr;
use crate::types::{DataType, Scalar};
use crate::hardware::HardwareEngine;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use pyo3::prelude::*;

static BROADCAST_ID_CTR: AtomicUsize = AtomicUsize::new(10_000);

#[derive(Debug, Clone)]
pub struct NumToyArray {
    pub expr: Expr,
        pub shape: Vec<usize>,
        pub bit_strides: Vec<usize>,
}

impl NumToyArray {
        pub fn dtype(&self) -> String {
        format!("{:?}", self.expr.data_type())
    }

        pub fn element_bits(&self) -> u32 {
        match self.expr.data_type() {
            DataType::Float(b) | DataType::Int(b) | DataType::ScalableInt(b) | DataType::ScalableFloat(b, _) => b,
            _ => 64,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.reverse();
        let mut new_strides = self.bit_strides.clone();
        new_strides.reverse();
        
        let new_expr = match &self.expr {
            Expr::Variable { id, name, dtype, size, packed_data, scale, steal_sign, bit_offset, .. } => {
                Expr::Variable {
                    id: *id,
                    name: name.clone(),
                    dtype: *dtype,
                    size: *size,
                    packed_data: packed_data.clone(),
                    shape: new_shape.clone(),
                    bit_strides: new_strides.clone(),
                    bit_offset: *bit_offset,
                    scale: *scale,
                    steal_sign: *steal_sign,
                }
            },
            _ => panic!("Can only transpose materialized variables currently"),
        };
        NumToyArray {
            expr: new_expr,
            shape: new_shape,
            bit_strides: new_strides,
        }
    }
    
    // TODO: slice_and_dice
}

impl NumToyArray {
    pub fn from_expr(expr: Expr, shape: Vec<usize>, bit_strides: Vec<usize>) -> Self {
        NumToyArray { expr, shape, bit_strides }
    }

    
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "NumToyArray::reshape - element count mismatch"
        );
        let strides = row_major_bit_strides(&new_shape, self.element_bits() as usize);
        NumToyArray {
            expr: self.expr.clone(),
            shape: new_shape,
            bit_strides: strides,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    // ─── Broadcast ─────────────────────────────────────────────────────────

    pub fn broadcast_to(&self, engine: &HardwareEngine, target_shape: Vec<usize>) -> Self {
        let src_shape = &self.shape;

        let rank = target_shape.len().max(src_shape.len());
        let mut src_padded = vec![1usize; rank];
        for (i, &d) in src_shape.iter().rev().enumerate() {
            src_padded[rank - 1 - i] = d;
        }
        for i in 0..rank {
            let sd = src_padded[i];
            let td = target_shape[i];
            assert!(
                sd == td || sd == 1,
                "NumToyArray::broadcast_to – incompatible shapes at dim {}: {} vs {}", i, sd, td
            );
        }

        let numel: usize = target_shape.iter().product();
        if numel == self.numel() && target_shape == *src_shape {
            return NumToyArray::from_expr(self.expr.clone(), target_shape, self.bit_strides.clone());
        }

        // Fast path for broadcasting a scalar or 1D over another
        let src_flat = self.to_flat_f64(engine);

        // Compute element-count row-major strides (NOT bit strides) for coordinate mapping.
        let tgt_elem_strides = row_major_element_strides(&target_shape);
        let mut src_elem_strides = row_major_element_strides(&src_padded);
        for i in 0..rank {
            if src_padded[i] == 1 {
                src_elem_strides[i] = 0;  // broadcast: stride=0 means always index element 0 on this axis
            }
        }

        let mut out = vec![0.0f64; numel];
        for flat_out in 0..numel {
            let mut src_flat_idx = 0usize;
            let mut remaining = flat_out;
            for dim in 0..rank {
                let stride = tgt_elem_strides[dim].max(1);
                let coord = remaining / stride;
                remaining %= stride;
                src_flat_idx += coord * src_elem_strides[dim];
            }
            out[flat_out] = src_flat[src_flat_idx.min(src_flat.len() - 1)];
        }

        let dtype = self.expr.data_type();
        let scale = self.expr.get_scale();
        let steal = out.iter().all(|&v| v >= 0.0)
            && matches!(dtype, crate::types::DataType::Float(_) | crate::types::DataType::DynamicFloat | crate::types::DataType::ScalableFloat(_, _));

        let u64s: Vec<u64> = out.iter().flat_map(|&f| {
            let scalar = match dtype {
                crate::types::DataType::Float(b) => Scalar::Float(f, b),
                crate::types::DataType::Int(b) => Scalar::Int(f.round() as i64, b),
                crate::types::DataType::DynamicFloat => Scalar::DynamicFloat(f),
                crate::types::DataType::FloatingInt =>
                    crate::types::double_to_floating_int(f, scale.unwrap_or(1)),
                crate::types::DataType::ScalableInt(b) => Scalar::ScalableInt(f.round() as i64, b),
                crate::types::DataType::ScalableFloat(b, e) => Scalar::ScalableFloat(f, b, e),
            };
            scalar.to_limbs(steal)
        }).collect();

        let bit_width = match dtype {
            crate::types::DataType::Float(b) | crate::types::DataType::Int(b) => b,
            crate::types::DataType::ScalableInt(b) | crate::types::DataType::ScalableFloat(b, _) => b,
            _ => 64,
        };
        let packed = engine.pack(&u64s, bit_width);
        let tiled_expr = Expr::new_var(
            BROADCAST_ID_CTR.fetch_add(1, Ordering::Relaxed),
            "broadcast",
            dtype,
            numel,
            crate::ir::PackedBuffer::Memory(std::sync::Arc::new(packed)),
            target_shape.clone(),
            row_major_bit_strides(&target_shape, bit_width as usize),
            0,
            scale,
            steal,
        );
        NumToyArray::from_expr(tiled_expr, target_shape.clone(), row_major_bit_strides(&target_shape, bit_width as usize))
    }

    fn align_broadcast(
        &self,
        other: &NumToyArray,
        engine: &HardwareEngine,
    ) -> (NumToyArray, NumToyArray, Vec<usize>) {
        let out_shape = broadcast_shape(&self.shape, &other.shape);
        let a = self.broadcast_to(engine, out_shape.clone());
        let b = other.broadcast_to(engine, out_shape.clone());
        let a = if !matches!(a.expr, Expr::Variable { .. }) {
            a.execute(engine, "cpu")
        } else { a };
        let b = if !matches!(b.expr, Expr::Variable { .. }) {
            b.execute(engine, "cpu")
        } else { b };
        (a, b, out_shape)
    }

    pub fn add(&self, other: &NumToyArray, engine: &HardwareEngine) -> NumToyArray {
        let (a, b, shape) = self.align_broadcast(other, engine);
        NumToyArray::from_expr(a.expr.clone().add(b.expr.clone()), shape.clone(), row_major_bit_strides(&shape, a.element_bits() as usize))
    }

    pub fn sub(&self, other: &NumToyArray, engine: &HardwareEngine) -> NumToyArray {
        let (a, b, shape) = self.align_broadcast(other, engine);
        NumToyArray::from_expr(a.expr.clone().sub(b.expr.clone()), shape.clone(), row_major_bit_strides(&shape, a.element_bits() as usize))
    }

    pub fn mul(&self, other: &NumToyArray, engine: &HardwareEngine) -> NumToyArray {
        let (a, b, shape) = self.align_broadcast(other, engine);
        NumToyArray::from_expr(a.expr.clone().mul(b.expr.clone()), shape.clone(), row_major_bit_strides(&shape, a.element_bits() as usize))
    }

    pub fn div(&self, other: &NumToyArray, engine: &HardwareEngine) -> NumToyArray {
        let (a, b, shape) = self.align_broadcast(other, engine);
        NumToyArray::from_expr(a.expr.clone().div(b.expr.clone()), shape.clone(), row_major_bit_strides(&shape, a.element_bits() as usize))
    }

    pub fn grad(&self, wrt_id: usize) -> NumToyArray {
        NumToyArray::from_expr(self.expr.grad(wrt_id), self.shape.clone(), self.bit_strides.clone())
    }

    pub fn execute(&self, engine: &HardwareEngine, _device: &str) -> NumToyArray {
        let result_expr = execute_expr_on_device(engine, &self.expr, _device);
        NumToyArray::from_expr(result_expr, self.shape.clone(), self.bit_strides.clone())
    }

    pub fn to_flat_f64(&self, engine: &HardwareEngine) -> Vec<f64> {
        match &self.expr {
            Expr::Variable { dtype, size, packed_data, scale, steal_sign, .. } => {
                let bit_width = match dtype {
                    DataType::Float(b) => *b,
                    DataType::Int(b) => *b,
                    DataType::DynamicFloat => 64,
                    DataType::FloatingInt => 64,
                    DataType::ScalableInt(b) => *b,
                    DataType::ScalableFloat(b, _) => *b,
                };
                let raw = engine.unpack(packed_data.as_slice(), *size, bit_width);
                let k = ((bit_width + 63) / 64) as usize;
                raw.chunks_exact(k)
                    .map(|limbs| Scalar::from_limbs(limbs, *dtype, *scale, *steal_sign).to_double())
                    .collect()
            }
            _ => panic!("NumToyArray::to_flat_f64 – call execute() first"),
        }
    }

}

pub fn row_major_bit_strides(shape: &[usize], bits_per_element: usize) -> Vec<usize> {
    let mut strides = vec![bits_per_element; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Row-major element strides (stride in number of elements, not bits).
/// E.g. for shape [3, 2]: strides = [2, 1].
pub fn row_major_element_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn execute_expr_on_device(_engine: &HardwareEngine, expr: &Expr, _device: &str) -> Expr {
    let mut graph = crate::graph::ArenaGraph::from_expr(expr);
    
    let mode = crate::get_execution_mode();
    if mode == crate::ExecutionMode::Eager {
        graph = (*crate::calculator::GLOBAL_CALCULATOR.submit_and_wait(std::sync::Arc::new(graph))).clone();
    }

    if _device == "gpu" {
        return crate::gpu::execute_gpu(_engine, expr, &graph);
    }
    let raw_dtype = expr.data_type();
    let target_scale = expr.get_scale();
    let target_steal = expr.steal_sign();
    
    // ScalableInt is widened to Int(64) inside the JIT to prevent overflow.
    let target_dtype = match raw_dtype {
        DataType::ScalableInt(_) => DataType::Int(64),
        other => other,
    };

    let kernel = crate::fuser::MicroKernel::generate(expr);
    let out_bits = match target_dtype {
        DataType::Float(bits) | DataType::Int(bits) => bits,
        DataType::DynamicFloat | DataType::FloatingInt => 64,
        DataType::ScalableInt(bits) | DataType::ScalableFloat(bits, _) => bits,
    };
    
    let out_bytes = if out_bits % 8 == 0 {
        kernel.size * (out_bits as usize / 8)
    } else {
        (kernel.size * out_bits as usize + 7) / 8
    };
    let mut output_buf = vec![0u8; out_bytes];
    
    let run_fn = if mode == crate::ExecutionMode::Eager {
        // Eager bypasses the pool and compiles synchronously
        let hash = crate::cache::hash_graph(&graph);
        if let Some(exec_fn) = crate::cache::JIT_CACHE.get(&hash) {
            *exec_fn
        } else {
            let exec_fn = crate::fuser::compile_packed_kernel(&graph).expect("Eager compilation failed");
            crate::cache::JIT_CACHE.insert(hash, exec_fn);
            exec_fn
        }
    } else {
        let hash = crate::pool::GLOBAL_POOL.submit(graph);
        loop {
            if let Some(f) = crate::cache::JIT_CACHE.get(&hash) {
                break *f;
            }
            std::thread::yield_now();
        }
    };
    
    // Build input pointers in the same order the JIT compiler assigned input_indices:
    // deduplicate by variable id, preserving first-occurrence order.
    let mut seen_ids: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut ordered_inputs: Vec<&Expr> = Vec::new();
    for var in &kernel.inputs {
        if let Expr::Variable { id, .. } = var {
            if seen_ids.insert(*id) {
                ordered_inputs.push(var);
            }
        }
    }
    let input_ptrs: Vec<*const u8> = ordered_inputs.iter().map(|input_var| {
        match input_var {
            Expr::Variable { packed_data, .. } => packed_data.as_slice().as_ptr(),
            _ => unreachable!(),
        }
    }).collect();
    
    unsafe { run_fn(input_ptrs.as_ptr(), output_buf.as_mut_ptr(), kernel.size); }
    
    let output_arc = Arc::new(output_buf);
    let shape = vec![kernel.size];
    let bit_strides = row_major_bit_strides(&shape, out_bits as usize);
    
    let result_id = BROADCAST_ID_CTR.fetch_add(1, Ordering::Relaxed);
    Expr::new_var(
        result_id,
        "fused_pool_result",
        target_dtype,
        kernel.size,
        crate::ir::PackedBuffer::Memory(output_arc),
        shape,
        bit_strides,
        0,
        target_scale,
        target_steal,
    )
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let ai = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let bi = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        out[rank - 1 - i] = if ai == bi {
            ai
        } else if ai == 1 {
            bi
        } else if bi == 1 {
            ai
        } else {
            panic!("broadcast_shape – incompatible dimensions {}, {}", ai, bi)
        };
    }
    out
}
