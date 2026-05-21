/// Unified Memory Tensor
///
/// A multi-dimensional array backed by a packed-byte `Expr` variable
/// plus shape and stride metadata.  All arithmetic produces new `Tensor`
/// values (lazy evaluation); call `.execute()` to JIT-compile and run.

use crate::ir::Expr;
use crate::types::{DataType, Scalar};
use crate::hardware::HardwareEngine;
use crate::fuser::execute_expr_on_device;
use std::sync::atomic::{AtomicUsize, Ordering};

static BROADCAST_ID_CTR: AtomicUsize = AtomicUsize::new(10_000);

#[derive(Debug, Clone)]
pub struct Tensor {
    /// The flat IR expression backing this tensor.
    pub expr: Expr,
    /// Shape in major→minor order, e.g. `[rows, cols]`.
    pub shape: Vec<usize>,
    /// Row-major strides in *elements* (not bytes).
    pub strides: Vec<usize>,
}

// ─── Construction ──────────────────────────────────────────────────────────

impl Tensor {
    /// Wrap an existing flat `Expr` with shape information.
    pub fn from_expr(expr: Expr, shape: Vec<usize>) -> Self {
        let strides = row_major_strides(&shape);
        Tensor { expr, shape, strides }
    }

    /// Total number of elements (product of shape dimensions).
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    // ─── Reshape ───────────────────────────────────────────────────────────

    /// Return a new `Tensor` with a different shape over the same data.
    /// Total element count must remain the same.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Tensor::reshape – element count mismatch: {} → {}",
            self.numel(),
            new_numel,
        );
        let strides = row_major_strides(&new_shape);
        Tensor {
            expr: self.expr.clone(),
            shape: new_shape,
            strides,
        }
    }

    // ─── Broadcast ─────────────────────────────────────────────────────────

    /// Broadcast `self` to `target_shape` using NumPy-compatible broadcasting rules.
    ///
    /// Dimensions are aligned from the right; a dim of size 1 is expanded by
    /// computing each output element's source index via modulo on each axis.
    pub fn broadcast_to(&self, engine: &HardwareEngine, target_shape: Vec<usize>) -> Self {
        let src_shape = &self.shape;

        // Validate and build aligned src shape (pad left with 1s)
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
                "Tensor::broadcast_to – incompatible shapes at dim {}: {} vs {}", i, sd, td
            );
        }

        let numel: usize = target_shape.iter().product();
        if numel == self.numel() && target_shape == *src_shape {
            return Tensor::from_expr(self.expr.clone(), target_shape);
        }

        // Unpack source to flat f64
        let src_flat = self.to_flat_f64(engine);

        // Compute target strides and source strides (with broadcast 0-stride for dim=1)
        let tgt_strides = row_major_strides(&target_shape);
        let mut src_strides = row_major_strides(&src_padded);
        for i in 0..rank {
            if src_padded[i] == 1 {
                src_strides[i] = 0; // broadcast: always index element 0 on this axis
            }
        }

        // Fill output by mapping each flat output index → source index
        let mut out = vec![0.0f64; numel];
        for flat_out in 0..numel {
            let mut src_flat_idx = 0usize;
            let mut remaining = flat_out;
            for dim in 0..rank {
                let coord = remaining / tgt_strides[dim];
                remaining %= tgt_strides[dim];
                src_flat_idx += coord * src_strides[dim];
            }
            out[flat_out] = src_flat[src_flat_idx];
        }

        // Repack as the same dtype/bits as the source
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
        let unique_id = BROADCAST_ID_CTR.fetch_add(1, Ordering::Relaxed);
        let tiled_expr = Expr::new_var(unique_id, "broadcast", dtype, numel, packed, scale, steal);
        Tensor::from_expr(tiled_expr, target_shape)
    }

    // ─── Element-wise arithmetic (with implicit broadcast) ─────────────────

    pub fn add(&self, other: &Tensor, engine: &HardwareEngine) -> Tensor {
        let (a, b, shape) = self.align_broadcast(other, engine);
        Tensor::from_expr(a.expr.add(b.expr), shape)
    }

    pub fn sub(&self, other: &Tensor, engine: &HardwareEngine) -> Tensor {
        let (a, b, shape) = self.align_broadcast(other, engine);
        Tensor::from_expr(a.expr.sub(b.expr), shape)
    }

    pub fn mul(&self, other: &Tensor, engine: &HardwareEngine) -> Tensor {
        let (a, b, shape) = self.align_broadcast(other, engine);
        Tensor::from_expr(a.expr.mul(b.expr), shape)
    }

    pub fn div(&self, other: &Tensor, engine: &HardwareEngine) -> Tensor {
        let (a, b, shape) = self.align_broadcast(other, engine);
        Tensor::from_expr(a.expr.div(b.expr), shape)
    }

    // ─── Auto-diff ─────────────────────────────────────────────────────────

    /// Return the symbolic gradient tensor d(self) / d(wrt_id).
    pub fn grad(&self, wrt_id: usize) -> Tensor {
        Tensor::from_expr(self.expr.grad(wrt_id), self.shape.clone())
    }

    // ─── Execution ─────────────────────────────────────────────────────────

    /// JIT-compile and execute the backing expression graph.
    pub fn execute(&self, engine: &HardwareEngine, device: &str) -> Tensor {
        let result_expr = execute_expr_on_device(engine, &self.expr, device);
        Tensor::from_expr(result_expr, self.shape.clone())
    }

    // ─── Unpack to flat f64 vector ─────────────────────────────────────────

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
                let raw = engine.unpack(packed_data, *size, bit_width);
                let k = ((bit_width + 63) / 64) as usize;
                raw.chunks_exact(k)
                    .map(|limbs| Scalar::from_limbs(limbs, *dtype, *scale, *steal_sign).to_double())
                    .collect()
            }
            _ => panic!("Tensor::to_flat_f64 – call execute() first"),
        }
    }

    // ─── Helpers ───────────────────────────────────────────────────────────

    fn packed_bytes(&self) -> Vec<u8> {
        match &self.expr {
            Expr::Variable { packed_data, .. } => packed_data.clone(),
            _ => panic!("Tensor::packed_bytes – tensor is not a variable"),
        }
    }

    fn align_broadcast(
        &self,
        other: &Tensor,
        engine: &HardwareEngine,
    ) -> (Tensor, Tensor, Vec<usize>) {
        let out_shape = broadcast_shape(&self.shape, &other.shape);
        // Materialize each broadcast result into a concrete Variable so
        // MicroKernel sees two distinct variables with unique IDs.
        let a = self.broadcast_to(engine, out_shape.clone());
        let b = other.broadcast_to(engine, out_shape.clone());
        // Execute each materialized broadcast so they become true Variables
        let a = if !matches!(a.expr, Expr::Variable { .. }) {
            a.execute(engine, "cpu")
        } else { a };
        let b = if !matches!(b.expr, Expr::Variable { .. }) {
            b.execute(engine, "cpu")
        } else { b };
        (a, b, out_shape)
    }
}

// ─── Shape utilities ───────────────────────────────────────────────────────

/// Compute row-major strides for a given shape.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute the output shape from two shapes under NumPy-style broadcasting.
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
            panic!(
                "broadcast_shape – incompatible dimensions {} and {}",
                ai, bi
            )
        };
    }
    out
}
