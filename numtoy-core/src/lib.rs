pub mod types;
pub mod hardware;
pub mod ir;
pub mod fuser;

use pyo3::prelude::*;
use crate::types::{DataType, Scalar};
use crate::hardware::HardwareEngine;
use crate::ir::Expr;
use crate::fuser::execute_expr_on_device;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────
// AdaptableFloat Sign Stealer analysis
// Checks whether all values in a float dataset are non-negative.
// If yes, the sign bit is discarded and shifted to the mantissa.
// ─────────────────────────────────────────────────────────
fn should_steal_sign(dtype: DataType, values: &[f64]) -> bool {
    match dtype {
        DataType::Float(_) | DataType::DynamicFloat => values.iter().all(|&v| v >= 0.0),
        _ => false, // Int & FloatingInt have their own sign representations
    }
}

// ─────────────────────────────────────────────────────────
// Python bindings
// ─────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: Expr,
}

#[pyclass]
pub struct PyEngine {
    pub inner: Arc<HardwareEngine>,
}

#[pymethods]
impl PyEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(HardwareEngine::new()),
        }
    }
}

#[pymethods]
impl PyExpr {
    #[staticmethod]
    pub fn new_var(
        engine: &PyEngine,
        id: usize,
        name: String,
        dtype_str: &str,
        bits: u32,
        values: Vec<f64>,
        scale: Option<u32>,
    ) -> Self {
        let dtype = match dtype_str {
            "float" => DataType::Float(bits),
            "int" => DataType::Int(bits),
            "dynamic_float" => DataType::DynamicFloat,
            "floating_int" => DataType::FloatingInt,
            _ => panic!("Unknown datatype: {}", dtype_str),
        };

        // ── AdaptableFloat Sign Stealer check ──
        let steal_sign = should_steal_sign(dtype, &values);
        if steal_sign {
            eprintln!("[NumToy] Sign Stealer activated for '{}': all values non-negative, \
                       sign bit reallocated to mantissa for +1 bit of precision", name);
        }

        let scalars: Vec<Scalar> = values
            .iter()
            .map(|&v| match dtype {
                DataType::Float(b) => Scalar::Float(v, b),
                DataType::Int(b) => Scalar::Int(v.round() as i64, b),
                DataType::DynamicFloat => Scalar::DynamicFloat(v),
                DataType::FloatingInt => crate::types::double_to_floating_int(v, scale.unwrap_or(1)),
            })
            .collect();

        let u64s: Vec<u64> = scalars.iter().map(|s| s.to_u64_packed(steal_sign)).collect();
        let bit_width = match dtype {
            DataType::Float(b) => b,
            DataType::Int(b) => b,
            DataType::DynamicFloat => 64,
            DataType::FloatingInt => 64,
        };
        let packed = engine.inner.pack(&u64s, bit_width);

        Self {
            inner: Expr::new_var(id, &name, dtype, values.len(), packed, scale, steal_sign),
        }
    }

    #[staticmethod]
    pub fn new_const(value: f64) -> Self {
        Self {
            inner: Expr::new_const(Scalar::DynamicFloat(value)),
        }
    }

    pub fn add(&self, other: &PyExpr) -> Self {
        Self {
            inner: self.inner.clone().add(other.inner.clone()),
        }
    }

    pub fn mul(&self, other: &PyExpr) -> Self {
        Self {
            inner: self.inner.clone().mul(other.inner.clone()),
        }
    }

    #[pyo3(signature = (engine, device = "cpu"))]
    pub fn execute(&self, engine: &PyEngine, device: &str) -> PyExpr {
        let result = execute_expr_on_device(&engine.inner, &self.inner, device);
        PyExpr { inner: result }
    }

    pub fn unpack(&self, engine: &PyEngine) -> Vec<f64> {
        match &self.inner {
            Expr::Variable { dtype, size, packed_data, scale, steal_sign, .. } => {
                let bit_width = match dtype {
                    DataType::Float(b) => *b,
                    DataType::Int(b) => *b,
                    DataType::DynamicFloat => 64,
                    DataType::FloatingInt => 64,
                };
                let raw_u64s = engine.inner.unpack(packed_data, *size, bit_width);
                raw_u64s
                    .into_iter()
                    .map(|u| Scalar::from_u64_packed(u, *dtype, *scale, *steal_sign).to_double())
                    .collect()
            }
            _ => panic!("Cannot unpack non-variable expression"),
        }
    }
}

#[pymodule]
fn numtoy_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyExpr>()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────
// C-ABI exports for C++ bindings
// ─────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn nt_engine_new() -> *mut HardwareEngine {
    Box::into_raw(Box::new(HardwareEngine::new()))
}

#[no_mangle]
pub unsafe extern "C" fn nt_engine_free(engine: *mut HardwareEngine) {
    if !engine.is_null() {
        let _ = Box::from_raw(engine);
    }
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_new_var(
    engine: *mut HardwareEngine,
    id: usize,
    name: *const libc::c_char,
    dtype_val: u32, // 0 = Float, 1 = Int, 2 = DynamicFloat, 3 = FloatingInt
    bits: u32,
    values: *const f64,
    count: usize,
    scale: u32,
) -> *mut Expr {
    let engine_ref = &*engine;
    let name_str = if name.is_null() {
        "var"
    } else {
        std::ffi::CStr::from_ptr(name).to_str().unwrap_or("var")
    };

    let dtype = match dtype_val {
        0 => DataType::Float(bits),
        1 => DataType::Int(bits),
        2 => DataType::DynamicFloat,
        3 => DataType::FloatingInt,
        _ => DataType::DynamicFloat,
    };

    let slice = std::slice::from_raw_parts(values, count);

    // ── AdaptableFloat Sign Stealer check ──
    let steal_sign = should_steal_sign(dtype, slice);

    let scalars: Vec<Scalar> = slice
        .iter()
        .map(|&v| match dtype {
            DataType::Float(b) => Scalar::Float(v, b),
            DataType::Int(b) => Scalar::Int(v.round() as i64, b),
            DataType::DynamicFloat => Scalar::DynamicFloat(v),
            DataType::FloatingInt => crate::types::double_to_floating_int(v, scale),
        })
        .collect();

    let u64s: Vec<u64> = scalars.iter().map(|s| s.to_u64_packed(steal_sign)).collect();
    let bit_width = match dtype {
        DataType::Float(b) => b,
        DataType::Int(b) => b,
        DataType::DynamicFloat => 64,
        DataType::FloatingInt => 64,
    };
    let packed = engine_ref.pack(&u64s, bit_width);

    let expr = Expr::new_var(id, name_str, dtype, count, packed, Some(scale), steal_sign);
    Box::into_raw(Box::new(expr))
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_new_const(val: f64) -> *mut Expr {
    let expr = Expr::new_const(Scalar::DynamicFloat(val));
    Box::into_raw(Box::new(expr))
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_add(left: *mut Expr, right: *mut Expr) -> *mut Expr {
    let l = (*left).clone();
    let r = (*right).clone();
    let expr = l.add(r);
    Box::into_raw(Box::new(expr))
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_mul(left: *mut Expr, right: *mut Expr) -> *mut Expr {
    let l = (*left).clone();
    let r = (*right).clone();
    let expr = l.mul(r);
    Box::into_raw(Box::new(expr))
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_free(expr: *mut Expr) {
    if !expr.is_null() {
        let _ = Box::from_raw(expr);
    }
}

/// device_str: null or "cpu" for CPU JIT, "gpu" for WebGPU
#[no_mangle]
pub unsafe extern "C" fn nt_expr_execute(
    engine: *mut HardwareEngine,
    expr: *mut Expr,
    device_str: *const libc::c_char,
) -> *mut Expr {
    let engine_ref = &*engine;
    let expr_ref = &*expr;
    let device = if device_str.is_null() {
        "cpu"
    } else {
        std::ffi::CStr::from_ptr(device_str).to_str().unwrap_or("cpu")
    };
    let result = execute_expr_on_device(engine_ref, expr_ref, device);
    Box::into_raw(Box::new(result))
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_unpack(
    engine: *mut HardwareEngine,
    expr: *mut Expr,
    out_values: *mut f64,
    max_count: usize,
) -> usize {
    let engine_ref = &*engine;
    let expr_ref = &*expr;
    match expr_ref {
        Expr::Variable { dtype, size, packed_data, scale, steal_sign, .. } => {
            let bit_width = match dtype {
                DataType::Float(b) => *b,
                DataType::Int(b) => *b,
                DataType::DynamicFloat => 64,
                DataType::FloatingInt => 64,
            };
            let raw_u64s = engine_ref.unpack(packed_data, *size, bit_width);
            let count = std::cmp::min(*size, max_count);
            for i in 0..count {
                *out_values.add(i) = Scalar::from_u64_packed(raw_u64s[i], *dtype, *scale, *steal_sign).to_double();
            }
            count
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuser::execute_expr_on_device;

    #[test]
    fn test_end_to_end_jit_compilation() {
        let engine = HardwareEngine::new();

        let x_vals = vec![1.0f64, 2.0, 3.0];
        let y_vals = vec![4.0f64, 5.0, 6.0];

        // All positive → sign stealer activates for Float32
        let steal = true;

        let x_scalars: Vec<Scalar> = x_vals.iter().map(|&v| Scalar::Float(v, 32)).collect();
        let y_scalars: Vec<Scalar> = y_vals.iter().map(|&v| Scalar::Float(v, 32)).collect();

        let x_u64s: Vec<u64> = x_scalars.iter().map(|s| s.to_u64_packed(steal)).collect();
        let y_u64s: Vec<u64> = y_scalars.iter().map(|s| s.to_u64_packed(steal)).collect();

        let x_packed = engine.pack(&x_u64s, 32);
        let y_packed = engine.pack(&y_u64s, 32);

        let x_expr = Expr::new_var(1, "x", DataType::Float(32), 3, x_packed, None, steal);
        let y_expr = Expr::new_var(2, "y", DataType::Float(32), 3, y_packed, None, steal);

        let const_expr = Expr::new_const(Scalar::DynamicFloat(2.0));
        let add_expr = x_expr.add(y_expr);
        let z_expr = add_expr.mul(const_expr);

        let result_expr = execute_expr_on_device(&engine, &z_expr, "cpu");

        match result_expr {
            Expr::Variable { dtype, size, packed_data, scale, steal_sign, .. } => {
                assert_eq!(dtype, DataType::Float(32));
                assert_eq!(size, 3);
                let raw_u64s = engine.unpack(&packed_data, size, 32);
                let results: Vec<f64> = raw_u64s
                    .into_iter()
                    .map(|u| Scalar::from_u64_packed(u, dtype, scale, steal_sign).to_double())
                    .collect();
                // With custom float encoding, allow small floating point error
                for (got, expected) in results.iter().zip([10.0, 14.0, 18.0].iter()) {
                    let err = (got - expected).abs();
                    assert!(err < 0.1, "got {} expected {} (err {})", got, expected, err);
                }
            }
            _ => panic!("Expected Variable result"),
        }
    }
}
