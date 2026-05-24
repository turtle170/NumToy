pub mod types;
pub mod hardware;
pub mod ir;
pub mod fuser;
pub mod array;
pub mod graph;
pub mod cache;
pub mod pool;
pub mod gpu;
pub mod calculator;
pub mod jit;
pub mod signals;
pub mod network;

use pyo3::prelude::*;
use crate::types::{DataType, Scalar};
use crate::hardware::HardwareEngine;
use crate::ir::Expr;
use crate::array::execute_expr_on_device;
use crate::array::NumToyArray;
use std::sync::Arc;

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Default = 0,
    Eager   = 1,
    /// Hyper: `THREAD_PRIORITY_TIME_CRITICAL` on Windows; crossbeam pool spins before yielding.
    Hyper   = 2,
    /// Xtreme: everything Hyper does **plus** —
    ///   • `REALTIME_PRIORITY_CLASS` for the whole process (Windows)
    ///   • Worker threads spin *forever* — no yield, no sleep
    ///   • All threads steal from all peer queues (not just assemblers)
    ///   • `Tile → Assemble` directly, bypassing the Fuse optimization pass
    ///   • Submit skips the JIT-cache probe; the Assemble handler deduplicates
    ///   • `_mm_prefetch` on graph nodes before JIT compilation
    ///   • Cranelift alias analysis enabled for better kernel code
    Xtreme  = 3,
}

pub static GLOBAL_EXECUTION_MODE: AtomicU8 = AtomicU8::new(0);

pub fn get_execution_mode() -> ExecutionMode {
    match GLOBAL_EXECUTION_MODE.load(Ordering::Relaxed) {
        1 => ExecutionMode::Eager,
        2 => ExecutionMode::Hyper,
        3 => ExecutionMode::Xtreme,
        _ => ExecutionMode::Default,
    }
}

pub fn set_execution_mode(mode: ExecutionMode) {
    let is_xtreme = mode == ExecutionMode::Xtreme;

    GLOBAL_EXECUTION_MODE.store(mode as u8, Ordering::Relaxed);

    // Flip the fast-path bool used in every worker hot-loop iteration.
    crate::pool::XTREME_ACTIVE.store(is_xtreme, Ordering::Relaxed);

    // In Xtreme mode, elevate the *entire process* to REALTIME_PRIORITY_CLASS
    // (Windows). This places the process above nearly every other process,
    // including many driver threads — the caller accepts that trade-off.
    // Exiting Xtreme mode resets to NORMAL_PRIORITY_CLASS.
    #[cfg(windows)]
    {
        use windows::Win32::System::Threading::{
            GetCurrentProcess, SetPriorityClass,
            REALTIME_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS,
        };
        unsafe {
            let _ = SetPriorityClass(
                GetCurrentProcess(),
                if is_xtreme { REALTIME_PRIORITY_CLASS } else { NORMAL_PRIORITY_CLASS },
            );
        }
    }

    crate::pool::update_pool_mode();
}

// ─────────────────────────────────────────────────────────
// AdaptableFloat Sign Stealer analysis
// ─────────────────────────────────────────────────────────
fn should_steal_sign(dtype: DataType, values: &[f64]) -> bool {
    match dtype {
        DataType::Float(_) | DataType::DynamicFloat | DataType::ScalableFloat(_, _) => values.iter().all(|&v| v >= 0.0),
        _ => false,
    }
}

#[pyfunction]
pub fn set_execution_mode_py(mode: &str) {
    let m = match mode {
        "eager"  => ExecutionMode::Eager,
        "hyper"  => ExecutionMode::Hyper,
        "xtreme" => ExecutionMode::Xtreme,
        _        => ExecutionMode::Default,
    };
    set_execution_mode(m);
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

        Self { inner: Expr::new_var(id, &name, dtype, values.len(), crate::ir::PackedBuffer::Memory(std::sync::Arc::new(packed)), vec![values.len()], crate::array::row_major_bit_strides(&[values.len()], dtype.bit_width() as usize), 0, scale, steal_sign) }
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
                let raw = engine.inner.unpack(packed_data.as_slice(), *size, bit_width);
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

#[pyclass(name="NumToyArray")]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: crate::array::NumToyArray,
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
            DataType::DynamicFloat | DataType::FloatingInt => 64,
        };
        let packed = engine.inner.pack(&u64s, bit_width);
        let expr = Expr::new_var(id, &name, dtype, values.len(), crate::ir::PackedBuffer::Memory(std::sync::Arc::new(packed)), vec![values.len()], crate::array::row_major_bit_strides(&[values.len()], dtype.bit_width() as usize), 0, scale, steal_sign);
        Self { inner: crate::array::NumToyArray::from_expr(expr.clone(), shape.clone(), crate::array::row_major_bit_strides(&shape, expr.data_type().bit_width() as usize)) }
    }

    #[staticmethod]
    pub fn from_mmap(
        id: usize,
        name: String,
        filepath: &str,
        count: usize,
        shape: Vec<usize>,
        dtype_str: &str,
        bits: u32,
        scale: Option<u32>,
    ) -> PyResult<Self> {
        let dtype = match dtype_str {
            "float"         => DataType::Float(bits),
            "int"           => DataType::Int(bits),
            "dynamic_float" => DataType::DynamicFloat,
            "floating_int"  => DataType::FloatingInt,
            "scalable_int"  => DataType::ScalableInt(bits),
            "scalable_float"=> DataType::ScalableFloat(bits, 4), // Placeholder exponent bits
            _               => panic!("Unknown datatype: {}", dtype_str),
        };

        let file = std::fs::File::open(filepath)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        
        let steal_sign = false; // mmap loaded arrays cannot compute dynamic sign-stealing trivially
        let expr = Expr::new_var(id, &name, dtype, count, crate::ir::PackedBuffer::Mmap(std::sync::Arc::new(mmap)), vec![count], crate::array::row_major_bit_strides(&[count], dtype.bit_width() as usize), 0, scale, steal_sign);
        Ok(Self { inner: crate::array::NumToyArray::from_expr(expr.clone(), shape.clone(), crate::array::row_major_bit_strides(&shape, expr.data_type().bit_width() as usize)) })
    }

    #[staticmethod]
    pub fn dequantize_matmul(
        activations: &PyTensor,
        weights: &PyTensor,
    ) -> Self {
        let mut shape = vec![1, 1];
        let mut m = 1;
        let mut k = 1;
        let mut n = 1;
        if activations.inner.shape.len() >= 2 && weights.inner.shape.len() >= 2 {
            m = activations.inner.shape[0];
            k = activations.inner.shape[1];
            n = weights.inner.shape[1];
            shape = vec![m, n];
        }
        let new_expr = Expr::DequantizeMatmul {
            activations: std::sync::Arc::new(activations.inner.expr.clone()),
            weights: std::sync::Arc::new(weights.inner.expr.clone()),
            m,
            k,
            n,
        };
        let strides = crate::array::row_major_bit_strides(&shape, 64); // output is f64
        Self { inner: crate::array::NumToyArray::from_expr(new_expr, shape, strides) }
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

    #[getter]
    pub fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);
        
        let shape_tuple = pyo3::types::PyTuple::new(py, &self.inner.shape).unwrap();
        dict.set_item("shape", shape_tuple)?;
        
        let mut byte_strides = Vec::new();
        for s in &self.inner.bit_strides {
            if s % 8 != 0 {
                return Err(pyo3::exceptions::PyValueError::new_err("Cannot expose non-byte-aligned strides via __array_interface__"));
            }
            byte_strides.push(s / 8);
        }
        let strides_tuple = pyo3::types::PyTuple::new(py, &byte_strides).unwrap();
        dict.set_item("strides", strides_tuple)?;
        
        let typestr = match self.inner.expr.data_type() {
            crate::types::DataType::Float(32) => "<f4",
            crate::types::DataType::Float(64) => "<f8",
            crate::types::DataType::Int(8) => "<i1",
            crate::types::DataType::Int(16) => "<i2",
            crate::types::DataType::Int(32) => "<i4",
            crate::types::DataType::Int(64) => "<i8",
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Cannot expose custom bit-types, fixed-point, or scalable types via zero-copy __array_interface__")),
        };
        dict.set_item("typestr", typestr)?;
        dict.set_item("version", 3)?;
        
        match &self.inner.expr {
            crate::ir::Expr::Variable { packed_data, bit_offset, steal_sign, .. } => {
                if *steal_sign {
                    return Err(pyo3::exceptions::PyValueError::new_err("Cannot expose sign-stolen AdaptableFloat arrays via zero-copy __array_interface__"));
                }
                if bit_offset % 8 != 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err("Cannot expose non-byte-aligned offset via __array_interface__"));
                }
                let ptr = packed_data.as_slice().as_ptr() as usize + (bit_offset / 8);
                let data_tuple = pyo3::types::PyTuple::new(py, vec![ptr, 0]).unwrap();
                dict.set_item("data", data_tuple)?; // tuple of pointer and readonly flag
            }
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Cannot expose unmaterialized array. Call execute() first.")),
        }
        
        Ok(dict)
    }

    pub fn slice_and_dice(&self, ranges: Vec<(usize, usize, usize)>) -> Self {
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        
        let (packed_data, mut bit_offset, scale, steal_sign, dtype, id, name) = match &self.inner.expr {
            crate::ir::Expr::Variable { packed_data, bit_offset, scale, steal_sign, dtype, id, name, .. } => {
                (packed_data.clone(), *bit_offset, *scale, *steal_sign, *dtype, *id, name.clone())
            }
            _ => panic!("Can only slice materialized variables currently"),
        };

        for (i, &(start, stop, step)) in ranges.iter().enumerate() {
            let dim_size = self.inner.shape[i];
            let actual_stop = stop.min(dim_size);
            let span = actual_stop.saturating_sub(start);
            if span > 0 {
                let elements = (span + step - 1) / step;
                new_shape.push(elements);
                new_strides.push(self.inner.bit_strides[i] * step);
                bit_offset += start * self.inner.bit_strides[i];
            } else {
                new_shape.push(0);
                new_strides.push(self.inner.bit_strides[i]);
            }
        }
        
        for i in ranges.len()..self.inner.shape.len() {
            new_shape.push(self.inner.shape[i]);
            new_strides.push(self.inner.bit_strides[i]);
        }
        
        let new_expr = crate::ir::Expr::Variable {
            id,
            name,
            dtype,
            size: new_shape.iter().product(),
            packed_data,
            shape: new_shape.clone(),
            bit_strides: new_strides.clone(),
            bit_offset,
            scale,
            steal_sign,
        };
        
        Self { inner: crate::array::NumToyArray::from_expr(new_expr, new_shape, new_strides) }
    }

    pub fn transpose(&self) -> Self {
        Self { inner: self.inner.transpose() }
    }

}

// ─────────────────────────────────────────────────────────
// Python module
// ─────────────────────────────────────────────────────────

#[pymodule]
fn numtoy_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    signals::install();
    m.add_class::<PyEngine>()?;
    m.add_class::<PyExpr>()?;
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(set_execution_mode_py, m)?)?;
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
    let expr = Expr::new_var(id, name_str, dtype, count, crate::ir::PackedBuffer::Memory(std::sync::Arc::new(packed)), vec![count], crate::array::row_major_bit_strides(&[count], dtype.bit_width() as usize), 0, Some(scale), steal_sign);
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
            let raw = (*engine).unpack(packed_data.as_slice(), *size, bit_width);
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
    inner: crate::array::NumToyArray,
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
    let tensor = crate::array::NumToyArray::from_expr(expr.clone(), shape.clone(), crate::array::row_major_bit_strides(&shape, expr.data_type().bit_width() as usize));
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
    use crate::array::execute_expr_on_device;

    fn make_var(engine: &HardwareEngine, id: usize, values: &[f64]) -> Expr {
        let steal = values.iter().all(|&v| v >= 0.0);
        let scalars: Vec<Scalar> = values.iter().map(|&v| Scalar::Float(v, 32)).collect();
        let u64s: Vec<u64> = scalars.iter().flat_map(|s| s.to_limbs(steal)).collect();
        let packed = engine.pack(&u64s, 32);
        Expr::new_var(id, &format!("v{id}"), DataType::Float(32), values.len(), crate::ir::PackedBuffer::Memory(std::sync::Arc::new(packed)), vec![values.len()], crate::array::row_major_bit_strides(&[values.len()], DataType::Float(32).bit_width() as usize), 0, None, steal)
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
                let raw = engine.unpack(packed_data.as_slice(), size, 32);
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
                let raw = engine.unpack(packed_data.as_slice(), size, 32);
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
        let a = crate::array::NumToyArray::from_expr(
            a_expr.clone(),
            vec![3, 1],
            crate::array::row_major_bit_strides(&[3, 1], a_expr.data_type().bit_width() as usize),
        );
        let b = crate::array::NumToyArray::from_expr(
            b_expr.clone(),
            vec![1, 2],
            crate::array::row_major_bit_strides(&[1, 2], b_expr.data_type().bit_width() as usize),
        );
        let c = a.add(&b, &engine).execute(&engine, "cpu");
        let flat = c.to_flat_f64(&engine);
        assert_eq!(flat.len(), 6);
        let expected = [11.0, 21.0, 12.0, 22.0, 13.0, 23.0];
        for (got, exp) in flat.iter().zip(expected) {
            assert!((got - exp).abs() < 0.5, "got {} expected {}", got, exp);
        }
    }
}
