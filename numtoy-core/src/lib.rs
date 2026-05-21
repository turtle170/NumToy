pub mod types;
pub mod hardware;
pub mod ir;
pub mod fuser;
pub mod tensor;
pub mod graph;
pub mod cache;
pub mod pool;

use pyo3::prelude::*;
use crate::types::{DataType, Scalar};
use crate::hardware::HardwareEngine;
use crate::ir::Expr;
use crate::tensor::execute_expr_on_device;
use crate::tensor::Tensor;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────
// AdaptableFloat Sign Stealer analysis
// ─────────────────────────────────────────────────────────
fn should_steal_sign(dtype: DataType, values: &[f64]) -> bool {
    match dtype {
        DataType::Float(_) | DataType::DynamicFloat | DataType::ScalableFloat(_, _) => values.iter().all(|&v| v >= 0.0),
        _ => false,
    }
}

// ─────────────────────────────────────────────────────────
// Python: PyEngine
// ─────────────────────────────────────────────────────────

#[pyclass]
pub struct PyEngine {
    pub inner: Arc<HardwareEngine>,
}

#[pymethods]
impl PyEngine {
    #[new]
    pub fn new() -> Self {
        Self { inner: Arc::new(HardwareEngine::new()) }
    }
}

// ─────────────────────────────────────────────────────────
// Python: PyExpr
// ─────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: Expr,
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
            "float"        => DataType::Float(bits),
            "int"          => DataType::Int(bits),
            "dynamic_float"=> DataType::DynamicFloat,
            "floating_int" => DataType::FloatingInt,
            "scalable_int" => {
                let max_bits = values.iter()
                    .map(|&v| crate::types::compute_scalable_int_bits(v.round() as i64))
                    .max()
                    .unwrap_or(2);
                DataType::ScalableInt(max_bits)
            }
            "scalable_float" => {
                let steal_sign = values.iter().all(|&v| v >= 0.0);
                let mut max_b = 8;
                let mut max_e = 4;
                for &v in &values {
                    let (b, e) = crate::types::compute_scalable_float_config(v, steal_sign);
                    if b > max_b { max_b = b; }
                    if e > max_e { max_e = e; }
                }
                let sign_bit = if steal_sign { 0 } else { 1 };
                if max_b < max_e + sign_bit + 1 {
                    max_b = max_e + sign_bit + 1;
                }
                if max_b > 64 {
                    max_b = 64;
                }
                DataType::ScalableFloat(max_b, max_e)
            }
            _              => panic!("Unknown datatype: {}", dtype_str),
        };

        let steal_sign = should_steal_sign(dtype, &values);
        if steal_sign {
            eprintln!("[NumToy] Sign Stealer activated for '{}': all values non-negative, \
                       sign bit reallocated to mantissa for +1 bit of precision", name);
        }

        let scalars: Vec<Scalar> = values.iter().map(|&v| match dtype {
            DataType::Float(b)  => Scalar::Float(v, b),
            DataType::Int(b)    => Scalar::Int(v.round() as i64, b),
            DataType::DynamicFloat => Scalar::DynamicFloat(v),
            DataType::FloatingInt  => crate::types::double_to_floating_int(v, scale.unwrap_or(1)),
            DataType::ScalableInt(b) => Scalar::ScalableInt(v.round() as i64, b),
            DataType::ScalableFloat(b, e) => Scalar::ScalableFloat(v, b, e),
        }).collect();

        let u64s: Vec<u64> = scalars.iter().flat_map(|s| s.to_limbs(steal_sign)).collect();
        let bit_width = match dtype {
            DataType::Float(b)  => b,
            DataType::Int(b)    => b,
            DataType::DynamicFloat | DataType::FloatingInt => 64,
            DataType::ScalableInt(b) | DataType::ScalableFloat(b, _) => b,
        };
        let packed = engine.inner.pack(&u64s, bit_width);

        Self { inner: Expr::new_var(id, &name, dtype, values.len(), packed, scale, steal_sign) }
    }

    #[staticmethod]
    pub fn new_const(value: f64) -> Self {
        Self { inner: Expr::new_const(Scalar::DynamicFloat(value)) }
    }

    pub fn add(&self, other: &PyExpr) -> Self {
        Self { inner: self.inner.clone().add(other.inner.clone()) }
    }

    pub fn sub(&self, other: &PyExpr) -> Self {
        Self { inner: self.inner.clone().sub(other.inner.clone()) }
    }

    pub fn mul(&self, other: &PyExpr) -> Self {
        Self { inner: self.inner.clone().mul(other.inner.clone()) }
    }

    pub fn div(&self, other: &PyExpr) -> Self {
        Self { inner: self.inner.clone().div(other.inner.clone()) }
    }

    /// Symbolic gradient: returns d(self)/d(var_id) as a new PyExpr.
    pub fn grad(&self, wrt_id: usize) -> Self {
        Self { inner: self.inner.grad(wrt_id) }
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
                    DataType::Int(b)   => *b,
                    DataType::DynamicFloat | DataType::FloatingInt => 64,
                    DataType::ScalableInt(b) => *b,
                    DataType::ScalableFloat(b, _) => *b,
                };
                let raw = engine.inner.unpack(packed_data, *size, bit_width);
                let k = ((bit_width + 63) / 64) as usize;
                raw.chunks_exact(k)
                    .map(|limbs| Scalar::from_limbs(limbs, *dtype, *scale, *steal_sign).to_double())
                    .collect()
            }
            _ => panic!("Cannot unpack non-variable expression; call execute() first"),
        }
    }
}

// ─────────────────────────────────────────────────────────
// Python: PyTensor
// ─────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: Tensor,
}

#[pymethods]
impl PyTensor {
    #[staticmethod]
    pub fn from_values(
        engine: &PyEngine,
        id: usize,
        name: String,
        values: Vec<f64>,
        shape: Vec<usize>,
        dtype_str: &str,
        bits: u32,
        scale: Option<u32>,
    ) -> Self {
        let dtype = match dtype_str {
            "float"         => DataType::Float(bits),
            "int"           => DataType::Int(bits),
            "dynamic_float" => DataType::DynamicFloat,
            "floating_int"  => DataType::FloatingInt,
            "scalable_int" => {
                let max_bits = values.iter()
                    .map(|&v| crate::types::compute_scalable_int_bits(v.round() as i64))
                    .max()
                    .unwrap_or(2);
                DataType::ScalableInt(max_bits)
            }
            "scalable_float" => {
                let steal_sign = values.iter().all(|&v| v >= 0.0);
                let mut max_b = 8;
                let mut max_e = 4;
                for &v in &values {
                    let (b, e) = crate::types::compute_scalable_float_config(v, steal_sign);
                    if b > max_b { max_b = b; }
                    if e > max_e { max_e = e; }
                }
                let sign_bit = if steal_sign { 0 } else { 1 };
                if max_b < max_e + sign_bit + 1 {
                    max_b = max_e + sign_bit + 1;
                }
                if max_b > 64 {
                    max_b = 64;
                }
                DataType::ScalableFloat(max_b, max_e)
            }
            _               => panic!("Unknown datatype: {}", dtype_str),
        };

        let steal_sign = should_steal_sign(dtype, &values);
        let scalars: Vec<Scalar> = values.iter().map(|&v| match dtype {
            DataType::Float(b)  => Scalar::Float(v, b),
            DataType::Int(b)    => Scalar::Int(v.round() as i64, b),
            DataType::DynamicFloat => Scalar::DynamicFloat(v),
            DataType::FloatingInt  => crate::types::double_to_floating_int(v, scale.unwrap_or(1)),
            DataType::ScalableInt(b) => Scalar::ScalableInt(v.round() as i64, b),
            DataType::ScalableFloat(b, e) => Scalar::ScalableFloat(v, b, e),
        }).collect();

        let u64s: Vec<u64> = scalars.iter().flat_map(|s| s.to_limbs(steal_sign)).collect();
        let bit_width = match dtype {
            DataType::Float(b) | DataType::Int(b) => b,
            DataType::ScalableInt(b) | DataType::ScalableFloat(b, _) => b,
            _ => 64,
        };
        let packed = engine.inner.pack(&u64s, bit_width);
        let expr = Expr::new_var(id, &name, dtype, values.len(), packed, scale, steal_sign);
        Self { inner: Tensor::from_expr(expr, shape) }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        Self { inner: self.inner.reshape(new_shape) }
    }

    pub fn broadcast_to(&self, engine: &PyEngine, target_shape: Vec<usize>) -> Self {
        Self { inner: self.inner.broadcast_to(&engine.inner, target_shape) }
    }

    pub fn add_tensor(&self, other: &PyTensor, engine: &PyEngine) -> Self {
        Self { inner: self.inner.add(&other.inner, &engine.inner) }
    }

    pub fn sub_tensor(&self, other: &PyTensor, engine: &PyEngine) -> Self {
        Self { inner: self.inner.sub(&other.inner, &engine.inner) }
    }

    pub fn mul_tensor(&self, other: &PyTensor, engine: &PyEngine) -> Self {
        Self { inner: self.inner.mul(&other.inner, &engine.inner) }
    }

    pub fn div_tensor(&self, other: &PyTensor, engine: &PyEngine) -> Self {
        Self { inner: self.inner.div(&other.inner, &engine.inner) }
    }

    pub fn grad(&self, wrt_id: usize) -> Self {
        Self { inner: self.inner.grad(wrt_id) }
    }

    #[pyo3(signature = (engine, device = "cpu"))]
    pub fn execute(&self, engine: &PyEngine, device: &str) -> Self {
        Self { inner: self.inner.execute(&engine.inner, device) }
    }

    pub fn to_flat_f64(&self, engine: &PyEngine) -> Vec<f64> {
        self.inner.to_flat_f64(&engine.inner)
    }
}

// ─────────────────────────────────────────────────────────
// Python module
// ─────────────────────────────────────────────────────────

#[pymodule]
fn numtoy_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyExpr>()?;
    m.add_class::<PyTensor>()?;
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
    if !engine.is_null() { let _ = Box::from_raw(engine); }
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_new_var(
    engine: *mut HardwareEngine,
    id: usize,
    name: *const libc::c_char,
    dtype_val: u32,
    bits: u32,
    values: *const f64,
    count: usize,
    scale: u32,
) -> *mut Expr {
    let engine_ref = &*engine;
    let name_str = if name.is_null() { "var" }
    else { std::ffi::CStr::from_ptr(name).to_str().unwrap_or("var") };

    let slice = std::slice::from_raw_parts(values, count);
    let dtype = match dtype_val {
        0 => DataType::Float(bits),
        1 => DataType::Int(bits),
        2 => DataType::DynamicFloat,
        3 => DataType::FloatingInt,
        4 => {
            let max_bits = slice.iter()
                .map(|&v| crate::types::compute_scalable_int_bits(v.round() as i64))
                .max()
                .unwrap_or(2);
            DataType::ScalableInt(max_bits)
        }
        5 => {
            let steal_sign = slice.iter().all(|&v| v >= 0.0);
            let mut max_b = 8;
            let mut max_e = 4;
            for &v in slice {
                let (b, e) = crate::types::compute_scalable_float_config(v, steal_sign);
                if b > max_b { max_b = b; }
                if e > max_e { max_e = e; }
            }
            let sign_bit = if steal_sign { 0 } else { 1 };
            if max_b < max_e + sign_bit + 1 {
                max_b = max_e + sign_bit + 1;
            }
            if max_b > 64 {
                max_b = 64;
            }
            DataType::ScalableFloat(max_b, max_e)
        }
        _ => DataType::DynamicFloat,
    };
    let steal_sign = should_steal_sign(dtype, slice);
    let scalars: Vec<Scalar> = slice.iter().map(|&v| match dtype {
        DataType::Float(b)  => Scalar::Float(v, b),
        DataType::Int(b)    => Scalar::Int(v.round() as i64, b),
        DataType::DynamicFloat => Scalar::DynamicFloat(v),
        DataType::FloatingInt  => crate::types::double_to_floating_int(v, scale),
        DataType::ScalableInt(b) => Scalar::ScalableInt(v.round() as i64, b),
        DataType::ScalableFloat(b, e) => Scalar::ScalableFloat(v, b, e),
    }).collect();
    let u64s: Vec<u64> = scalars.iter().flat_map(|s| s.to_limbs(steal_sign)).collect();
    let bit_width = match dtype {
        DataType::Float(b) | DataType::Int(b) => b,
        DataType::ScalableInt(b) | DataType::ScalableFloat(b, _) => b,
        _ => 64,
    };
    let packed = engine_ref.pack(&u64s, bit_width);
    let expr = Expr::new_var(id, name_str, dtype, count, packed, Some(scale), steal_sign);
    Box::into_raw(Box::new(expr))
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_new_const(val: f64) -> *mut Expr {
    Box::into_raw(Box::new(Expr::new_const(Scalar::DynamicFloat(val))))
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_add(left: *mut Expr, right: *mut Expr) -> *mut Expr {
    Box::into_raw(Box::new((*left).clone().add((*right).clone())))
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_sub(left: *mut Expr, right: *mut Expr) -> *mut Expr {
    Box::into_raw(Box::new((*left).clone().sub((*right).clone())))
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_mul(left: *mut Expr, right: *mut Expr) -> *mut Expr {
    Box::into_raw(Box::new((*left).clone().mul((*right).clone())))
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_div(left: *mut Expr, right: *mut Expr) -> *mut Expr {
    Box::into_raw(Box::new((*left).clone().div((*right).clone())))
}

/// Returns the symbolic gradient d(expr)/d(wrt_id) as a new Expr*.
#[no_mangle]
pub unsafe extern "C" fn nt_expr_grad(expr: *mut Expr, wrt_id: usize) -> *mut Expr {
    Box::into_raw(Box::new((*expr).grad(wrt_id)))
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_free(expr: *mut Expr) {
    if !expr.is_null() { let _ = Box::from_raw(expr); }
}

/// device_str: NULL/"cpu" for Cranelift JIT, "gpu" for WebGPU
#[no_mangle]
pub unsafe extern "C" fn nt_expr_execute(
    engine: *mut HardwareEngine,
    expr: *mut Expr,
    device_str: *const libc::c_char,
) -> *mut Expr {
    let device = if device_str.is_null() { "cpu" }
    else { std::ffi::CStr::from_ptr(device_str).to_str().unwrap_or("cpu") };
    let result = execute_expr_on_device(&*engine, &*expr, device);
    Box::into_raw(Box::new(result))
}

#[no_mangle]
pub unsafe extern "C" fn nt_expr_unpack(
    engine: *mut HardwareEngine,
    expr: *mut Expr,
    out_values: *mut f64,
    max_count: usize,
) -> usize {
    match &*expr {
        Expr::Variable { dtype, size, packed_data, scale, steal_sign, .. } => {
            let bit_width = match dtype {
                DataType::Float(b) | DataType::Int(b) => *b,
                DataType::ScalableInt(b) | DataType::ScalableFloat(b, _) => *b,
                _ => 64,
            };
            let raw = (*engine).unpack(packed_data, *size, bit_width);
            let count = std::cmp::min(*size, max_count);
            let k = ((bit_width + 63) / 64) as usize;
            for i in 0..count {
                let chunk = &raw[i * k .. (i + 1) * k];
                *out_values.add(i) =
                    Scalar::from_limbs(chunk, *dtype, *scale, *steal_sign).to_double();
            }
            count
        }
        _ => 0,
    }
}

// ─────────────────────────────────────────────────────────
// C-ABI: Tensor
// ─────────────────────────────────────────────────────────

#[repr(C)]
pub struct CppTensor {
    inner: Tensor,
}

#[no_mangle]
pub unsafe extern "C" fn nt_tensor_new(
    engine: *mut HardwareEngine,
    id: usize,
    name: *const libc::c_char,
    dtype_val: u32,
    bits: u32,
    values: *const f64,
    count: usize,
    scale: u32,
    shape_ptr: *const usize,
    shape_len: usize,
) -> *mut CppTensor {
    let expr_ptr = nt_expr_new_var(engine, id, name, dtype_val, bits, values, count, scale);
    if expr_ptr.is_null() { return std::ptr::null_mut(); }
    let expr = *Box::from_raw(expr_ptr);
    let shape = std::slice::from_raw_parts(shape_ptr, shape_len).to_vec();
    let tensor = Tensor::from_expr(expr, shape);
    Box::into_raw(Box::new(CppTensor { inner: tensor }))
}

#[no_mangle]
pub unsafe extern "C" fn nt_tensor_free(t: *mut CppTensor) {
    if !t.is_null() { let _ = Box::from_raw(t); }
}

#[no_mangle]
pub unsafe extern "C" fn nt_tensor_grad(t: *mut CppTensor, wrt_id: usize) -> *mut CppTensor {
    let grad_tensor = (*t).inner.grad(wrt_id);
    Box::into_raw(Box::new(CppTensor { inner: grad_tensor }))
}

#[no_mangle]
pub unsafe extern "C" fn nt_tensor_execute(
    engine: *mut HardwareEngine,
    t: *mut CppTensor,
    device_str: *const libc::c_char,
) -> *mut CppTensor {
    let device = if device_str.is_null() { "cpu" }
    else { std::ffi::CStr::from_ptr(device_str).to_str().unwrap_or("cpu") };
    let result = (*t).inner.execute(&*engine, device);
    Box::into_raw(Box::new(CppTensor { inner: result }))
}

// ─────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::execute_expr_on_device;

    fn make_var(engine: &HardwareEngine, id: usize, values: &[f64]) -> Expr {
        let steal = values.iter().all(|&v| v >= 0.0);
        let scalars: Vec<Scalar> = values.iter().map(|&v| Scalar::Float(v, 32)).collect();
        let u64s: Vec<u64> = scalars.iter().flat_map(|s| s.to_limbs(steal)).collect();
        let packed = engine.pack(&u64s, 32);
        Expr::new_var(id, &format!("v{id}"), DataType::Float(32), values.len(), packed, None, steal)
    }

    #[test]
    fn test_end_to_end_jit_compilation() {
        let engine = HardwareEngine::new();
        let x = make_var(&engine, 1, &[1.0, 2.0, 3.0]);
        let y = make_var(&engine, 2, &[4.0, 5.0, 6.0]);
        let z = x.add(y).mul(Expr::new_const(Scalar::DynamicFloat(2.0)));
        let result = execute_expr_on_device(&engine, &z, "cpu");
        match result {
            Expr::Variable { size, packed_data, dtype, scale, steal_sign, .. } => {
                assert_eq!(size, 3);
                let raw = engine.unpack(&packed_data, size, 32);
                let vals: Vec<f64> = raw.into_iter()
                    .map(|u| Scalar::from_u64_packed(u, dtype, scale, steal_sign).to_double())
                    .collect();
                for (got, exp) in vals.iter().zip([10.0, 14.0, 18.0]) {
                    assert!((got - exp).abs() < 0.1, "got {} expected {}", got, exp);
                }
            }
            _ => panic!("expected Variable"),
        }
    }

    #[test]
    fn test_auto_diff_product_rule() {
        // z = (x + y) * x   =>   dz/dx = 2*x + y
        // At x=[2.0], y=[3.0]: dz/dx = 2*2 + 3 = 7
        let engine = HardwareEngine::new();
        let x = make_var(&engine, 1, &[2.0]);
        let y = make_var(&engine, 2, &[3.0]);
        let z = (x.clone().add(y.clone())).mul(x.clone());
        let dz_dx = z.grad(1); // wrt x (id=1)
        let result = execute_expr_on_device(&engine, &dz_dx, "cpu");
        match result {
            Expr::Variable { size, packed_data, dtype, scale, steal_sign, .. } => {
                let raw = engine.unpack(&packed_data, size, 32);
                let val = Scalar::from_u64_packed(raw[0], dtype, scale, steal_sign).to_double();
                assert!((val - 7.0).abs() < 0.5, "d(z)/d(x) = {} expected 7.0", val);
            }
            _ => panic!("expected Variable"),
        }
    }

    #[test]
    fn test_tensor_broadcast_add() {
        // a = [[1.0], [2.0], [3.0]]  shape [3,1]
        // b = [[10.0, 20.0]]          shape [1,2]
        // a + b should be [[11,21],[12,22],[13,23]]  shape [3,2]
        let engine = HardwareEngine::new();
        let a_expr = make_var(&engine, 1, &[1.0, 2.0, 3.0]);
        let b_expr = make_var(&engine, 2, &[10.0, 20.0]);
        let a = Tensor::from_expr(a_expr, vec![3, 1]);
        let b = Tensor::from_expr(b_expr, vec![1, 2]);
        let c = a.add(&b, &engine).execute(&engine, "cpu");
        let flat = c.to_flat_f64(&engine);
        assert_eq!(flat.len(), 6);
        let expected = [11.0, 21.0, 12.0, 22.0, 13.0, 23.0];
        for (got, exp) in flat.iter().zip(expected) {
            assert!((got - exp).abs() < 0.5, "got {} expected {}", got, exp);
        }
    }
}
