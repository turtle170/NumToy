use crate::ir::Expr;
use crate::types::{DataType, Scalar, double_to_floating_int};
use crate::hardware::HardwareEngine;
use std::collections::HashMap;

use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::ir::{AbiParam, Value, Type, MemFlags, condcodes::IntCC, types as cl_types, InstBuilder};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

// ─────────────────────────────────────────────────────────
// Stage 1 – IR traversal & micro-kernel generation
// ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum FusedOp {
    Input(usize),
    ConstVal(f64),
    Add(Box<FusedOp>, Box<FusedOp>),
    Sub(Box<FusedOp>, Box<FusedOp>),
    Mul(Box<FusedOp>, Box<FusedOp>),
    Div(Box<FusedOp>, Box<FusedOp>),
}

pub struct MicroKernel {
    pub op: FusedOp,
    pub inputs: Vec<Expr>, // unique variables
    pub size: usize,
}

impl MicroKernel {
    pub fn generate(expr: &Expr) -> Self {
        let mut inputs = Vec::new();
        let mut var_map = HashMap::new();

        fn traverse(
            node: &Expr,
            inputs: &mut Vec<Expr>,
            var_map: &mut HashMap<usize, usize>,
        ) -> FusedOp {
            match node {
                Expr::Variable { id, .. } => {
                    let idx = *var_map.entry(*id).or_insert_with(|| {
                        inputs.push(node.clone());
                        inputs.len() - 1
                    });
                    FusedOp::Input(idx)
                }
                Expr::Constant { val } => FusedOp::ConstVal(val.to_double()),
                Expr::Add { left, right } => {
                    let l = traverse(left, inputs, var_map);
                    let r = traverse(right, inputs, var_map);
                    FusedOp::Add(Box::new(l), Box::new(r))
                }
                Expr::Sub { left, right } => {
                    let l = traverse(left, inputs, var_map);
                    let r = traverse(right, inputs, var_map);
                    FusedOp::Sub(Box::new(l), Box::new(r))
                }
                Expr::Mul { left, right } => {
                    let l = traverse(left, inputs, var_map);
                    let r = traverse(right, inputs, var_map);
                    FusedOp::Mul(Box::new(l), Box::new(r))
                }
                Expr::Div { left, right } => {
                    let l = traverse(left, inputs, var_map);
                    let r = traverse(right, inputs, var_map);
                    FusedOp::Div(Box::new(l), Box::new(r))
                }
            }
        }

        let op = traverse(expr, &mut inputs, &mut var_map);
        let size = expr.size();
        MicroKernel { op, inputs, size }
    }
}

// ─────────────────────────────────────────────────────────
// Stage 2 – CPU JIT via Cranelift
// ─────────────────────────────────────────────────────────

pub fn compile_kernel(kernel: &MicroKernel) -> Option<extern "C" fn(*const *const f64, *mut f64, usize)> {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").ok();
    flag_builder.set("opt_level", "speed").ok();
    
    let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
        panic!("host machine is not supported: {}", msg);
    });
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();
    let default_call_conv = isa.default_call_conv();
    let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jit_builder);

    let mut ctx = module.make_context();
    let ptr_type = module.target_config().pointer_type();
    ctx.func.signature.call_conv = default_call_conv;

    // fn(inputs: *const *const f64, output: *mut f64, len: usize)
    ctx.func.signature.params.push(AbiParam::new(ptr_type));
    ctx.func.signature.params.push(AbiParam::new(ptr_type));
    ctx.func.signature.params.push(AbiParam::new(ptr_type));

    let mut builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let inputs_arg = builder.block_params(entry_block)[0];
    let output_arg = builder.block_params(entry_block)[1];
    let len_arg = builder.block_params(entry_block)[2];

    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    let idx_var = builder.declare_var(ptr_type);
    let zero = builder.ins().iconst(ptr_type, 0);
    builder.def_var(idx_var, zero);

    builder.ins().jump(loop_header, &[]);

    // Loop Header
    builder.switch_to_block(loop_header);
    let i_val = builder.use_var(idx_var);
    let cmp = builder.ins().icmp(IntCC::UnsignedLessThan, i_val, len_arg);
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    // Loop Body
    builder.switch_to_block(loop_body);

    fn compile_op(
        builder: &mut FunctionBuilder,
        op: &FusedOp,
        inputs_arg: Value,
        i_val: Value,
        ptr_type: Type,
    ) -> Value {
        match op {
            FusedOp::Input(idx) => {
                let ptr_size = if ptr_type == cl_types::I64 { 8 } else { 4 };
                let offset = (*idx * ptr_size) as i32;
                let arr_ptr = builder.ins().load(ptr_type, MemFlags::new(), inputs_arg, offset);
                
                let scale = builder.ins().iconst(ptr_type, 8);
                let byte_offset = builder.ins().imul(i_val, scale);
                let addr = builder.ins().iadd(arr_ptr, byte_offset);
                builder.ins().load(cl_types::F64, MemFlags::new(), addr, 0)
            }
            FusedOp::ConstVal(val) => {
                builder.ins().f64const(*val)
            }
            FusedOp::Add(left, right) => {
                let l = compile_op(builder, left, inputs_arg, i_val, ptr_type);
                let r = compile_op(builder, right, inputs_arg, i_val, ptr_type);
                builder.ins().fadd(l, r)
            }
            FusedOp::Sub(left, right) => {
                let l = compile_op(builder, left, inputs_arg, i_val, ptr_type);
                let r = compile_op(builder, right, inputs_arg, i_val, ptr_type);
                builder.ins().fsub(l, r)
            }
            FusedOp::Mul(left, right) => {
                let l = compile_op(builder, left, inputs_arg, i_val, ptr_type);
                let r = compile_op(builder, right, inputs_arg, i_val, ptr_type);
                builder.ins().fmul(l, r)
            }
            FusedOp::Div(left, right) => {
                let l = compile_op(builder, left, inputs_arg, i_val, ptr_type);
                let r = compile_op(builder, right, inputs_arg, i_val, ptr_type);
                builder.ins().fdiv(l, r)
            }
        }
    }

    let result_val = compile_op(&mut builder, &kernel.op, inputs_arg, i_val, ptr_type);

    let scale = builder.ins().iconst(ptr_type, 8);
    let byte_offset = builder.ins().imul(i_val, scale);
    let out_addr = builder.ins().iadd(output_arg, byte_offset);
    builder.ins().store(MemFlags::new(), result_val, out_addr, 0);

    let one = builder.ins().iconst(ptr_type, 1);
    let next_i = builder.ins().iadd(i_val, one);
    builder.def_var(idx_var, next_i);

    builder.ins().jump(loop_header, &[]);

    builder.seal_block(loop_header);
    builder.seal_block(loop_body);
    builder.seal_block(loop_exit);

    builder.switch_to_block(loop_exit);
    builder.ins().return_(&[]);

    builder.finalize();

    let func_id = module.declare_function("run", Linkage::Export, &ctx.func.signature).ok()?;
    module.define_function(func_id, &mut ctx).ok()?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().ok()?;

    let code_ptr = module.get_finalized_function(func_id);
    let run_fn = unsafe { std::mem::transmute::<*const u8, extern "C" fn(*const *const f64, *mut f64, usize)>(code_ptr) };
    Some(run_fn)
}

// ─────────────────────────────────────────────────────────
// Stage 2 – GPU backend via wgpu (WebGPU / Vulkan / DX12)
// ─────────────────────────────────────────────────────────

/// Recursively build a WGSL expression string from the FusedOp tree.
fn wgsl_expr(op: &FusedOp) -> String {
    match op {
        FusedOp::Input(idx) => format!("inputs{}[idx]", idx),
        FusedOp::ConstVal(v) => {
            if v.fract() == 0.0 { format!("{:.1}", v) } else { format!("{}", v) }
        }
        FusedOp::Add(l, r) => format!("({} + {})", wgsl_expr(l), wgsl_expr(r)),
        FusedOp::Sub(l, r) => format!("({} - {})", wgsl_expr(l), wgsl_expr(r)),
        FusedOp::Mul(l, r) => format!("({} * {})", wgsl_expr(l), wgsl_expr(r)),
        FusedOp::Div(l, r) => format!("({} / {})", wgsl_expr(l), wgsl_expr(r)),
    }
}

/// Generate a complete WGSL compute shader for a given micro-kernel.
fn generate_wgsl(kernel: &MicroKernel) -> String {
    let n_inputs = kernel.inputs.len();
    let mut bindings = String::new();
    // Input buffers – binding 0..n_inputs
    for i in 0..n_inputs {
        bindings.push_str(&format!(
            "@group(0) @binding({i}) var<storage, read> inputs{i}: array<f32>;\n"
        ));
    }
    // Output buffer – binding n_inputs
    bindings.push_str(&format!(
        "@group(0) @binding({n_inputs}) var<storage, read_write> output: array<f32>;\n"
    ));

    // The compute expression operates on f32 in WGSL; we cast inputs from storage
    let expr_str = wgsl_expr(&kernel.op);

    format!(
        r#"
{bindings}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&output)) {{ return; }}
    output[idx] = {expr_str};
}}
"#,
        bindings = bindings,
        expr_str = expr_str
    )
}

/// Execute the fused kernel on the GPU using `wgpu`.
/// Input buffers are f64 host arrays; they are converted to f32 for the GPU
/// (WebGPU does not support native f64 in shaders), then results are promoted
/// back to f64.
pub fn execute_expr_gpu(engine: &HardwareEngine, expr: &Expr) -> Option<Expr> {
    if needs_limb_interpreter(expr) {
        return Some(execute_expr_limb_interpreter(engine, expr));
    }
    let kernel = MicroKernel::generate(expr);
    let steal_sign = expr.steal_sign();

    // ── Unpack inputs ─────────────────────────────────────
    let mut input_f32_bufs: Vec<Vec<f32>> = Vec::new();
    for input_var in &kernel.inputs {
        match input_var {
            Expr::Variable { dtype, size, packed_data, scale, steal_sign: var_ss, .. } => {
                let bit_width = match dtype {
                    DataType::Float(b) => *b,
                    DataType::Int(b) => *b,
                    DataType::DynamicFloat => 64,
                    DataType::FloatingInt => 64,
                    DataType::ScalableInt(b) => *b,
                    DataType::ScalableFloat(b, _) => *b,
                };
                let raw_u64s = engine.unpack(packed_data, *size, bit_width);
                let f32s: Vec<f32> = raw_u64s
                    .into_iter()
                    .map(|u| Scalar::from_u64_packed(u, *dtype, *scale, *var_ss).to_double() as f32)
                    .collect();
                input_f32_bufs.push(f32s);
            }
            _ => unreachable!(),
        }
    }

    // ── wgpu initialisation ───────────────────────────────
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("numtoy-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None, // trace path
    )).ok()?;

    // ── Build WGSL shader ─────────────────────────────────
    let wgsl = generate_wgsl(&kernel);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("numtoy_kernel"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    let size_bytes = (kernel.size * std::mem::size_of::<f32>()) as u64;

    // ── Upload input buffers ──────────────────────────────
    use wgpu::util::DeviceExt;
    let mut input_gpu_bufs: Vec<wgpu::Buffer> = Vec::new();
    for f32_buf in &input_f32_bufs {
        let bytes: &[u8] = bytemuck_cast_slice(f32_buf);
        let gpu_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        input_gpu_bufs.push(gpu_buf);
    }

    // ── Output buffer ─────────────────────────────────────
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Staging buffer for readback
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Bind group layout ─────────────────────────────────
    let n_inputs = kernel.inputs.len();
    let mut layout_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..n_inputs)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    layout_entries.push(wgpu::BindGroupLayoutEntry {
        binding: n_inputs as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("numtoy_bgl"),
        entries: &layout_entries,
    });

    // ── Bind group ────────────────────────────────────────
    let mut bind_group_entries: Vec<wgpu::BindGroupEntry> = input_gpu_bufs
        .iter()
        .enumerate()
        .map(|(i, buf)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.as_entire_binding(),
        })
        .collect();
    bind_group_entries.push(wgpu::BindGroupEntry {
        binding: n_inputs as u32,
        resource: output_buf.as_entire_binding(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("numtoy_bg"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    // ── Pipeline ──────────────────────────────────────────
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("numtoy_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("numtoy_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // ── Dispatch ──────────────────────────────────────────
    let workgroups = ((kernel.size as u32) + 63) / 64;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("numtoy_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("numtoy_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, size_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    // ── Readback ──────────────────────────────────────────
    let buf_slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buf_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().ok()?.ok()?;

    let output_f32: Vec<f32> = {
        let data = buf_slice.get_mapped_range();
        bytemuck_cast_slice_f32(&data).to_vec()
    };
    staging_buf.unmap();

    // ── Repack output as f64 → typed Expr ────────────────
    let target_dtype = expr.data_type();
    let target_scale = expr.get_scale();
    let target_bit_width = match target_dtype {
        DataType::Float(b) => b,
        DataType::Int(b) => b,
        DataType::DynamicFloat => 64,
        DataType::FloatingInt => 64,
        DataType::ScalableInt(b) => b,
        DataType::ScalableFloat(b, _) => b,
    };

    // Check sign steal for output (conservatively: all outputs non-negative)
    let output_steal = steal_sign && output_f32.iter().all(|&v| v >= 0.0);

    let output_u64s: Vec<u64> = output_f32
        .into_iter()
        .map(|f| {
            let fd = f as f64;
            let scalar = match target_dtype {
                DataType::Float(bits) => Scalar::Float(fd, bits),
                DataType::Int(bits) => Scalar::Int(fd.round() as i64, bits),
                DataType::DynamicFloat => Scalar::DynamicFloat(fd),
                DataType::FloatingInt => double_to_floating_int(fd, target_scale.unwrap_or(1)),
                DataType::ScalableInt(bits) => Scalar::ScalableInt(fd.round() as i64, bits),
                DataType::ScalableFloat(bits, e) => Scalar::ScalableFloat(fd, bits, e),
            };
            scalar.to_u64_packed(output_steal)
        })
        .collect();

    let packed_output = engine.pack(&output_u64s, target_bit_width);

    Some(Expr::new_var(
        999,
        "fused_gpu_result",
        target_dtype,
        kernel.size,
        packed_output,
        target_scale,
        output_steal,
    ))
}

/// Safe byte-cast helpers (avoids depending on `bytemuck` crate).
fn bytemuck_cast_slice(v: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            v.as_ptr() as *const u8,
            v.len() * std::mem::size_of::<f32>(),
        )
    }
}

fn bytemuck_cast_slice_f32(v: &[u8]) -> &[f32] {
    unsafe {
        std::slice::from_raw_parts(
            v.as_ptr() as *const f32,
            v.len() / std::mem::size_of::<f32>(),
        )
    }
}

// ─────────────────────────────────────────────────────────
// Secondary and Tertiary JIT Compiler-Grade Fusers
// ─────────────────────────────────────────────────────────

pub enum OpResult {
    Float(Value),
    Int64(Value),
    Int128(Value, Value),
}

fn collect_original_widths(node: &Expr, map: &mut HashMap<usize, u32>) {
    match node {
        Expr::Variable { id, dtype, .. } => {
            let bit_width = match dtype {
                DataType::Float(b) | DataType::Int(b) => *b,
                DataType::DynamicFloat | DataType::FloatingInt => 64,
                DataType::ScalableInt(b) | DataType::ScalableFloat(b, _) => *b,
            };
            map.insert(*id, bit_width);
        }
        Expr::Constant { .. } => {}
        Expr::Add { left, right } |
        Expr::Sub { left, right } |
        Expr::Mul { left, right } |
        Expr::Div { left, right } => {
            collect_original_widths(left, map);
            collect_original_widths(right, map);
        }
    }
}

pub extern "C" fn helper_load_u64(
    buf_ptr: *const u8,
    i: usize,
    bit_width: u32,
    limb_idx: u32,
) -> u64 {
    if bit_width == 0 {
        return 0;
    }
    match bit_width {
        8 => unsafe { *buf_ptr.add(i) as u64 },
        16 => unsafe {
            let ptr = buf_ptr.add(i * 2) as *const u16;
            ptr.read_unaligned() as u64
        },
        32 => unsafe {
            let ptr = buf_ptr.add(i * 4) as *const u32;
            ptr.read_unaligned() as u64
        },
        64 => unsafe {
            let ptr = buf_ptr.add(i * 8) as *const u64;
            ptr.read_unaligned()
        },
        _ if bit_width % 64 == 0 => unsafe {
            let offset = i * (bit_width as usize / 8) + (limb_idx as usize) * 8;
            let ptr = buf_ptr.add(offset) as *const u64;
            ptr.read_unaligned()
        },
        _ => {
            let start_bit = i * (bit_width as usize) + (limb_idx as usize) * 64;
            let mut val = 0u64;
            for bit_in_limb in 0..64 {
                if (limb_idx as usize) * 64 + bit_in_limb >= bit_width as usize {
                    break;
                }
                let current_bit = start_bit + bit_in_limb;
                let byte_idx = current_bit / 8;
                let bit_in_byte = current_bit % 8;
                let byte_val = unsafe { *buf_ptr.add(byte_idx) };
                let bit_val = (byte_val >> bit_in_byte) & 1;
                val |= (bit_val as u64) << bit_in_limb;
            }
            val
        }
    }
}

pub extern "C" fn helper_store_u64(
    buf_ptr: *mut u8,
    i: usize,
    bit_width: u32,
    limb_idx: u32,
    val: u64,
) {
    if bit_width == 0 {
        return;
    }
    match bit_width {
        8 => unsafe { *buf_ptr.add(i) = val as u8; },
        16 => unsafe {
            let ptr = buf_ptr.add(i * 2) as *mut u16;
            ptr.write_unaligned(val as u16);
        },
        32 => unsafe {
            let ptr = buf_ptr.add(i * 4) as *mut u32;
            ptr.write_unaligned(val as u32);
        },
        64 => unsafe {
            let ptr = buf_ptr.add(i * 8) as *mut u64;
            ptr.write_unaligned(val);
        },
        _ if bit_width % 64 == 0 => unsafe {
            let offset = i * (bit_width as usize / 8) + (limb_idx as usize) * 8;
            let ptr = buf_ptr.add(offset) as *mut u64;
            ptr.write_unaligned(val);
        },
        _ => {
            let start_bit = i * (bit_width as usize) + (limb_idx as usize) * 64;
            for bit_in_limb in 0..64 {
                if (limb_idx as usize) * 64 + bit_in_limb >= bit_width as usize {
                    break;
                }
                let current_bit = start_bit + bit_in_limb;
                let byte_idx = current_bit / 8;
                let bit_in_byte = current_bit % 8;
                let bit_val = ((val >> bit_in_limb) & 1) as u8;
                unsafe {
                    let byte_ptr = buf_ptr.add(byte_idx);
                    if bit_val == 1 {
                        *byte_ptr |= 1 << bit_in_byte;
                    } else {
                        *byte_ptr &= !(1 << bit_in_byte);
                    }
                }
            }
        }
    }
}

pub extern "C" fn helper_decode_to_f64(
    limb0: u64,
    limb1: u64,
    dtype_code: u32,
    bits: u32,
    e: u32,
    scale: u32,
    steal_sign_val: u8,
) -> f64 {
    let steal_sign = steal_sign_val != 0;
    let limbs = [limb0, limb1];
    let dtype = match dtype_code {
        0 => DataType::Float(bits),
        1 => DataType::Int(bits),
        2 => DataType::DynamicFloat,
        3 => DataType::FloatingInt,
        4 => DataType::ScalableInt(bits),
        5 => DataType::ScalableFloat(bits, e),
        _ => DataType::DynamicFloat,
    };
    Scalar::from_limbs(&limbs, dtype, Some(scale), steal_sign).to_double()
}

pub extern "C" fn helper_encode_from_f64(
    val: f64,
    dtype_code: u32,
    bits: u32,
    e: u32,
    scale: u32,
    steal_sign_val: u8,
    limb_idx: u32,
) -> u64 {
    let steal_sign = steal_sign_val != 0;
    let dtype = match dtype_code {
        0 => DataType::Float(bits),
        1 => DataType::Int(bits),
        2 => DataType::DynamicFloat,
        3 => DataType::FloatingInt,
        4 => DataType::ScalableInt(bits),
        5 => DataType::ScalableFloat(bits, e),
        _ => DataType::DynamicFloat,
    };
    let scalar = match dtype {
        DataType::Float(b) => Scalar::Float(val, b),
        DataType::Int(b) => {
            if b <= 64 {
                Scalar::Int(val.round() as i64, b)
            } else {
                Scalar::IntLimbs(crate::types::encode_custom_int_limbs_from_double(val, b), b)
            }
        }
        DataType::DynamicFloat => Scalar::DynamicFloat(val),
        DataType::FloatingInt => crate::types::double_to_floating_int(val, scale),
        DataType::ScalableInt(b) => {
            if b <= 64 {
                Scalar::ScalableInt(val.round() as i64, b)
            } else {
                Scalar::ScalableIntLimbs(crate::types::encode_custom_int_limbs_from_double(val, b), b)
            }
        }
        DataType::ScalableFloat(b, exp) => Scalar::ScalableFloat(val, b, exp),
    };
    let limbs = scalar.to_limbs(steal_sign);
    limbs.get(limb_idx as usize).copied().unwrap_or(0)
}

pub extern "C" fn helper_add_i128(l0: u64, l1: u64, r0: u64, r1: u64, limb: u32) -> u64 {
    let l = (l0 as u128) | ((l1 as u128) << 64);
    let r = (r0 as u128) | ((r1 as u128) << 64);
    let res = l.wrapping_add(r);
    if limb == 0 { res as u64 } else { (res >> 64) as u64 }
}

pub extern "C" fn helper_sub_i128(l0: u64, l1: u64, r0: u64, r1: u64, limb: u32) -> u64 {
    let l = (l0 as u128) | ((l1 as u128) << 64);
    let r = (r0 as u128) | ((r1 as u128) << 64);
    let res = l.wrapping_sub(r);
    if limb == 0 { res as u64 } else { (res >> 64) as u64 }
}

pub extern "C" fn helper_mul_i128(l0: u64, l1: u64, r0: u64, r1: u64, limb: u32) -> u64 {
    let l = (l0 as u128) | ((l1 as u128) << 64);
    let r = (r0 as u128) | ((r1 as u128) << 64);
    let res = l.wrapping_mul(r);
    if limb == 0 { res as u64 } else { (res >> 64) as u64 }
}

pub extern "C" fn helper_div_i128(l0: u64, l1: u64, r0: u64, r1: u64, limb: u32) -> u64 {
    let l = (l0 as u128) | ((l1 as u128) << 64);
    let r = (r0 as u128) | ((r1 as u128) << 64);
    let res = if r == 0 { 0 } else { l / r };
    if limb == 0 { res as u64 } else { (res >> 64) as u64 }
}

fn sign_extend_128(
    builder: &mut FunctionBuilder,
    low: Value,
    high: Value,
    current_bit_width: u32,
) -> (Value, Value) {
    if current_bit_width <= 64 {
        let shift_amt = 64 - current_bit_width;
        let shift_val = builder.ins().iconst(cl_types::I64, shift_amt as i64);
        let temp = builder.ins().ishl(low, shift_val);
        let low_extended = builder.ins().sshr(temp, shift_val);
        let shift_63 = builder.ins().iconst(cl_types::I64, 63);
        let high_extended = builder.ins().sshr(low_extended, shift_63);
        (low_extended, high_extended)
    } else if current_bit_width < 128 {
        let shift_amt = 128 - current_bit_width;
        let shift_val = builder.ins().iconst(cl_types::I64, shift_amt as i64);
        let temp = builder.ins().ishl(high, shift_val);
        let high_extended = builder.ins().sshr(temp, shift_val);
        (low, high_extended)
    } else {
        (low, high)
    }
}


fn compile_packed_op(
    builder: &mut FunctionBuilder,
    op: &FusedOp,
    inputs_arg: Value,
    i_val: Value,
    ptr_type: Type,
    is_float_mode: bool,
    max_bits: u32,
    local_func_load: cranelift_codegen::ir::FuncRef,
    local_func_decode: cranelift_codegen::ir::FuncRef,
    local_func_add_i128: cranelift_codegen::ir::FuncRef,
    local_func_sub_i128: cranelift_codegen::ir::FuncRef,
    local_func_mul_i128: cranelift_codegen::ir::FuncRef,
    local_func_div_i128: cranelift_codegen::ir::FuncRef,
    inputs: &[Expr],
    original_bit_widths: &HashMap<usize, u32>,
) -> OpResult {
    match op {
        FusedOp::Input(idx) => {
            let input_var = &inputs[*idx];
            let (id, dtype, scale, steal_sign) = match input_var {
                Expr::Variable { id, dtype, scale, steal_sign, .. } => (*id, *dtype, scale.unwrap_or(0), *steal_sign),
                _ => unreachable!(),
            };
            let current_bit_width = match dtype {
                DataType::Float(b) | DataType::Int(b) => b,
                DataType::DynamicFloat | DataType::FloatingInt => 64,
                DataType::ScalableInt(b) | DataType::ScalableFloat(b, _) => b,
            };
            let original_bit_width = *original_bit_widths.get(&id).unwrap_or(&current_bit_width);
            let e = match dtype {
                DataType::ScalableFloat(_, exp) => exp,
                DataType::Float(_) => if current_bit_width <= 16 { 5 } else if current_bit_width <= 32 { 8 } else { 11 },
                _ => 0,
            };
            
            let ptr_size = if ptr_type == cl_types::I64 { 8 } else { 4 };
            let offset = (*idx * ptr_size) as i32;
            let buf_ptr = builder.ins().load(ptr_type, MemFlags::new(), inputs_arg, offset);

            if is_float_mode {
                let k = ((original_bit_width + 63) / 64) as usize;
                let limb0 = {
                    let bit_width_val = builder.ins().iconst(cl_types::I32, original_bit_width as i64);
                    let limb_idx_val = builder.ins().iconst(cl_types::I32, 0);
                    let call = builder.ins().call(local_func_load, &[buf_ptr, i_val, bit_width_val, limb_idx_val]);
                    builder.inst_results(call)[0]
                };
                let limb1 = if k > 1 {
                    let bit_width_val = builder.ins().iconst(cl_types::I32, original_bit_width as i64);
                    let limb_idx_val = builder.ins().iconst(cl_types::I32, 1);
                    let call = builder.ins().call(local_func_load, &[buf_ptr, i_val, bit_width_val, limb_idx_val]);
                    builder.inst_results(call)[0]
                } else {
                    builder.ins().iconst(cl_types::I64, 0)
                };

                let dtype_code = match dtype {
                    DataType::Float(_) => 0,
                    DataType::Int(_) => 1,
                    DataType::DynamicFloat => 2,
                    DataType::FloatingInt => 3,
                    DataType::ScalableInt(_) => 4,
                    DataType::ScalableFloat(_, _) => 5,
                };
                let dtype_code_val = builder.ins().iconst(cl_types::I32, dtype_code);
                let bits_val = builder.ins().iconst(cl_types::I32, original_bit_width as i64);
                let e_val = builder.ins().iconst(cl_types::I32, e as i64);
                let scale_val = builder.ins().iconst(cl_types::I32, scale as i64);
                let steal_val = builder.ins().iconst(cl_types::I8, if steal_sign { 1 } else { 0 });

                let call_decode = builder.ins().call(
                    local_func_decode,
                    &[limb0, limb1, dtype_code_val, bits_val, e_val, scale_val, steal_val],
                );
                OpResult::Float(builder.inst_results(call_decode)[0])
            } else {
                if max_bits <= 64 {
                    let bit_width_val = builder.ins().iconst(cl_types::I32, original_bit_width as i64);
                    let limb_idx_val = builder.ins().iconst(cl_types::I32, 0);
                    let call = builder.ins().call(local_func_load, &[buf_ptr, i_val, bit_width_val, limb_idx_val]);
                    let raw_low = builder.inst_results(call)[0];
                    let low_extended = if current_bit_width < 64 {
                        let shift_amt = 64 - current_bit_width;
                        let shift_val = builder.ins().iconst(cl_types::I64, shift_amt as i64);
                        let temp = builder.ins().ishl(raw_low, shift_val);
                        builder.ins().sshr(temp, shift_val)
                    } else {
                        raw_low
                    };
                    OpResult::Int64(low_extended)
                } else {
                    let bit_width_val = builder.ins().iconst(cl_types::I32, original_bit_width as i64);
                    let limb0_idx = builder.ins().iconst(cl_types::I32, 0);
                    let call0 = builder.ins().call(local_func_load, &[buf_ptr, i_val, bit_width_val, limb0_idx]);
                    let raw_low = builder.inst_results(call0)[0];

                    let limb1_idx = builder.ins().iconst(cl_types::I32, 1);
                    let call1 = builder.ins().call(local_func_load, &[buf_ptr, i_val, bit_width_val, limb1_idx]);
                    let raw_high = builder.inst_results(call1)[0];

                    let (low_ext, high_ext) = sign_extend_128(builder, raw_low, raw_high, current_bit_width);
                    OpResult::Int128(low_ext, high_ext)
                }
            }
        }
        FusedOp::ConstVal(val) => {
            if is_float_mode {
                OpResult::Float(builder.ins().f64const(*val))
            } else {
                if max_bits <= 64 {
                    OpResult::Int64(builder.ins().iconst(cl_types::I64, *val as i64))
                } else {
                    let val_i128 = *val as i128;
                    let low = builder.ins().iconst(cl_types::I64, val_i128 as i64);
                    let high = builder.ins().iconst(cl_types::I64, (val_i128 >> 64) as i64);
                    OpResult::Int128(low, high)
                }
            }
        }
        FusedOp::Add(left, right) | FusedOp::Sub(left, right) | FusedOp::Mul(left, right) | FusedOp::Div(left, right) => {
            let l = compile_packed_op(builder, left, inputs_arg, i_val, ptr_type, is_float_mode, max_bits, local_func_load, local_func_decode, local_func_add_i128, local_func_sub_i128, local_func_mul_i128, local_func_div_i128, inputs, original_bit_widths);
            let r = compile_packed_op(builder, right, inputs_arg, i_val, ptr_type, is_float_mode, max_bits, local_func_load, local_func_decode, local_func_add_i128, local_func_sub_i128, local_func_mul_i128, local_func_div_i128, inputs, original_bit_widths);
            match (l, r) {
                (OpResult::Float(lf), OpResult::Float(rf)) => {
                    let res = match op {
                        FusedOp::Add(..) => builder.ins().fadd(lf, rf),
                        FusedOp::Sub(..) => builder.ins().fsub(lf, rf),
                        FusedOp::Mul(..) => builder.ins().fmul(lf, rf),
                        FusedOp::Div(..) => builder.ins().fdiv(lf, rf),
                        _ => unreachable!(),
                    };
                    OpResult::Float(res)
                }
                (OpResult::Int64(lv), OpResult::Int64(rv)) => {
                    let res = match op {
                        FusedOp::Add(..) => builder.ins().iadd(lv, rv),
                        FusedOp::Sub(..) => builder.ins().isub(lv, rv),
                        FusedOp::Mul(..) => builder.ins().imul(lv, rv),
                        FusedOp::Div(..) => builder.ins().sdiv(lv, rv),
                        _ => unreachable!(),
                    };
                    OpResult::Int64(res)
                }
                (OpResult::Int128(ll, lh), OpResult::Int128(rl, rh)) => {
                    let local_func = match op {
                        FusedOp::Add(..) => local_func_add_i128,
                        FusedOp::Sub(..) => local_func_sub_i128,
                        FusedOp::Mul(..) => local_func_mul_i128,
                        FusedOp::Div(..) => local_func_div_i128,
                        _ => unreachable!(),
                    };
                    let limb0_idx = builder.ins().iconst(cl_types::I32, 0);
                    let call_low = builder.ins().call(local_func, &[ll, lh, rl, rh, limb0_idx]);
                    let res_low = builder.inst_results(call_low)[0];

                    let limb1_idx = builder.ins().iconst(cl_types::I32, 1);
                    let call_high = builder.ins().call(local_func, &[ll, lh, rl, rh, limb1_idx]);
                    let res_high = builder.inst_results(call_high)[0];

                    OpResult::Int128(res_low, res_high)
                }
                _ => unreachable!(),
            }
        }
    }
}

pub fn compile_packed_kernel(
    kernel: &MicroKernel,
    is_float_mode: bool,
    compute_bits: u32,
    out_bits: u32,
    original_bit_widths: &HashMap<usize, u32>,
) -> Option<extern "C" fn(*const *const u8, *mut u8, usize)> {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").ok();
    flag_builder.set("opt_level", "speed").ok();
    
    let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
        panic!("host machine is not supported: {}", msg);
    });
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();
    let default_call_conv = isa.default_call_conv();
    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    
    jit_builder.symbol("helper_load_u64", helper_load_u64 as *const u8);
    jit_builder.symbol("helper_store_u64", helper_store_u64 as *const u8);
    jit_builder.symbol("helper_decode_to_f64", helper_decode_to_f64 as *const u8);
    jit_builder.symbol("helper_encode_from_f64", helper_encode_from_f64 as *const u8);
    jit_builder.symbol("helper_add_i128", helper_add_i128 as *const u8);
    jit_builder.symbol("helper_sub_i128", helper_sub_i128 as *const u8);
    jit_builder.symbol("helper_mul_i128", helper_mul_i128 as *const u8);
    jit_builder.symbol("helper_div_i128", helper_div_i128 as *const u8);

    let mut module = JITModule::new(jit_builder);
    let mut ctx = module.make_context();
    let ptr_type = module.target_config().pointer_type();
    ctx.func.signature.call_conv = default_call_conv;

    ctx.func.signature.params.push(AbiParam::new(ptr_type));
    ctx.func.signature.params.push(AbiParam::new(ptr_type));
    ctx.func.signature.params.push(AbiParam::new(ptr_type));

    let mut builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let inputs_arg = builder.block_params(entry_block)[0];
    let output_arg = builder.block_params(entry_block)[1];
    let len_arg = builder.block_params(entry_block)[2];

    let mut sig_load = module.make_signature();
    sig_load.params.push(AbiParam::new(ptr_type));
    sig_load.params.push(AbiParam::new(ptr_type));
    sig_load.params.push(AbiParam::new(cl_types::I32));
    sig_load.params.push(AbiParam::new(cl_types::I32));
    sig_load.returns.push(AbiParam::new(cl_types::I64));
    let func_load = module.declare_function("helper_load_u64", Linkage::Import, &sig_load).ok()?;
    let local_func_load = module.declare_func_in_func(func_load, builder.func);

    let mut sig_store = module.make_signature();
    sig_store.params.push(AbiParam::new(ptr_type));
    sig_store.params.push(AbiParam::new(ptr_type));
    sig_store.params.push(AbiParam::new(cl_types::I32));
    sig_store.params.push(AbiParam::new(cl_types::I32));
    sig_store.params.push(AbiParam::new(cl_types::I64));
    let func_store = module.declare_function("helper_store_u64", Linkage::Import, &sig_store).ok()?;
    let local_func_store = module.declare_func_in_func(func_store, builder.func);

    let mut sig_decode = module.make_signature();
    sig_decode.params.push(AbiParam::new(cl_types::I64));
    sig_decode.params.push(AbiParam::new(cl_types::I64));
    sig_decode.params.push(AbiParam::new(cl_types::I32));
    sig_decode.params.push(AbiParam::new(cl_types::I32));
    sig_decode.params.push(AbiParam::new(cl_types::I32));
    sig_decode.params.push(AbiParam::new(cl_types::I32));
    sig_decode.params.push(AbiParam::new(cl_types::I8));
    sig_decode.returns.push(AbiParam::new(cl_types::F64));
    let func_decode = module.declare_function("helper_decode_to_f64", Linkage::Import, &sig_decode).ok()?;
    let local_func_decode = module.declare_func_in_func(func_decode, builder.func);

    let mut sig_encode = module.make_signature();
    sig_encode.params.push(AbiParam::new(cl_types::F64));
    sig_encode.params.push(AbiParam::new(cl_types::I32));
    sig_encode.params.push(AbiParam::new(cl_types::I32));
    sig_encode.params.push(AbiParam::new(cl_types::I32));
    sig_encode.params.push(AbiParam::new(cl_types::I32));
    sig_encode.params.push(AbiParam::new(cl_types::I8));
    sig_encode.params.push(AbiParam::new(cl_types::I32));
    sig_encode.returns.push(AbiParam::new(cl_types::I64));
    let func_encode = module.declare_function("helper_encode_from_f64", Linkage::Import, &sig_encode).ok()?;
    let local_func_encode = module.declare_func_in_func(func_encode, builder.func);

    let mut sig_bin128 = module.make_signature();
    sig_bin128.params.push(AbiParam::new(cl_types::I64));
    sig_bin128.params.push(AbiParam::new(cl_types::I64));
    sig_bin128.params.push(AbiParam::new(cl_types::I64));
    sig_bin128.params.push(AbiParam::new(cl_types::I64));
    sig_bin128.params.push(AbiParam::new(cl_types::I32));
    sig_bin128.returns.push(AbiParam::new(cl_types::I64));
    let func_add_i128 = module.declare_function("helper_add_i128", Linkage::Import, &sig_bin128).ok()?;
    let local_func_add_i128 = module.declare_func_in_func(func_add_i128, builder.func);
    let func_sub_i128 = module.declare_function("helper_sub_i128", Linkage::Import, &sig_bin128).ok()?;
    let local_func_sub_i128 = module.declare_func_in_func(func_sub_i128, builder.func);
    let func_mul_i128 = module.declare_function("helper_mul_i128", Linkage::Import, &sig_bin128).ok()?;
    let local_func_mul_i128 = module.declare_func_in_func(func_mul_i128, builder.func);
    let func_div_i128 = module.declare_function("helper_div_i128", Linkage::Import, &sig_bin128).ok()?;
    let local_func_div_i128 = module.declare_func_in_func(func_div_i128, builder.func);

    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    let idx_var = builder.declare_var(ptr_type);
    let zero = builder.ins().iconst(ptr_type, 0);
    builder.def_var(idx_var, zero);

    builder.ins().jump(loop_header, &[]);

    builder.switch_to_block(loop_header);
    let i_val = builder.use_var(idx_var);
    let cmp = builder.ins().icmp(IntCC::UnsignedLessThan, i_val, len_arg);
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    builder.switch_to_block(loop_body);

    let result_val = compile_packed_op(
        &mut builder,
        &kernel.op,
        inputs_arg,
        i_val,
        ptr_type,
        is_float_mode,
        compute_bits,
        local_func_load,
        local_func_decode,
        local_func_add_i128,
        local_func_sub_i128,
        local_func_mul_i128,
        local_func_div_i128,
        &kernel.inputs,
        original_bit_widths,
    );

    let out_k = ((out_bits + 63) / 64) as usize;
    
    if is_float_mode {
        let f_val = match result_val {
            OpResult::Float(v) => v,
            _ => unreachable!(),
        };
        for limb_idx in 0..out_k {
            let dtype_code = 0; 
            let dtype_code_val = builder.ins().iconst(cl_types::I32, dtype_code);
            let bits_val = builder.ins().iconst(cl_types::I32, out_bits as i64);
            let e_val = builder.ins().iconst(cl_types::I32, 11); 
            let scale_val = builder.ins().iconst(cl_types::I32, 0);
            let steal_val = builder.ins().iconst(cl_types::I8, 0);
            let limb_idx_val = builder.ins().iconst(cl_types::I32, limb_idx as i64);
            let encoded_limb = builder.ins().call(
                local_func_encode,
                &[f_val, dtype_code_val, bits_val, e_val, scale_val, steal_val, limb_idx_val],
            );
            let encoded_limb_val = builder.inst_results(encoded_limb)[0];

            let out_bits_val = builder.ins().iconst(cl_types::I32, out_bits as i64);
            let out_limb_idx_val = builder.ins().iconst(cl_types::I32, limb_idx as i64);
            builder.ins().call(
                local_func_store,
                &[output_arg, i_val, out_bits_val, out_limb_idx_val, encoded_limb_val],
            );
        }
    } else {
        if out_bits <= 64 {
            let i_val_res = match result_val {
                OpResult::Int64(v) => v,
                OpResult::Int128(l, _) => l,
                _ => unreachable!(),
            };
            let out_bits_val = builder.ins().iconst(cl_types::I32, out_bits as i64);
            let out_limb_idx_val = builder.ins().iconst(cl_types::I32, 0);
            builder.ins().call(
                local_func_store,
                &[output_arg, i_val, out_bits_val, out_limb_idx_val, i_val_res],
            );
        } else {
            let (limb0, limb1) = match result_val {
                OpResult::Int128(l, h) => (l, h),
                OpResult::Int64(v) => {
                    let shift_amt = builder.ins().iconst(cl_types::I64, 63);
                    let h = builder.ins().sshr(v, shift_amt);
                    (v, h)
                },
                _ => unreachable!(),
            };
            let out_bits_val = builder.ins().iconst(cl_types::I32, out_bits as i64);
            let limb0_idx = builder.ins().iconst(cl_types::I32, 0);
            builder.ins().call(
                local_func_store,
                &[output_arg, i_val, out_bits_val, limb0_idx, limb0],
            );
            let limb1_idx = builder.ins().iconst(cl_types::I32, 1);
            builder.ins().call(
                local_func_store,
                &[output_arg, i_val, out_bits_val, limb1_idx, limb1],
            );
        }
    }

    let one = builder.ins().iconst(ptr_type, 1);
    let next_i = builder.ins().iadd(i_val, one);
    builder.def_var(idx_var, next_i);

    builder.ins().jump(loop_header, &[]);

    builder.seal_block(loop_header);
    builder.seal_block(loop_body);
    builder.seal_block(loop_exit);

    builder.switch_to_block(loop_exit);
    builder.ins().return_(&[]);

    builder.finalize();

    let func_id = module.declare_function("run_packed", Linkage::Export, &ctx.func.signature).ok()?;
    module.define_function(func_id, &mut ctx).ok()?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().ok()?;

    let code_ptr = module.get_finalized_function(func_id);
    let run_fn = unsafe {
        std::mem::transmute::<*const u8, extern "C" fn(*const *const u8, *mut u8, usize)>(code_ptr)
    };
    Some(run_fn)
}

pub fn execute_expr_secondary(engine: &HardwareEngine, expr: &Expr) -> Expr {
    execute_expr_secondary_internal(engine, expr, None)
}

fn execute_expr_secondary_internal(engine: &HardwareEngine, expr: &Expr, override_compute_bits: Option<u32>) -> Expr {
    let kernel = MicroKernel::generate(expr);
    let target_dtype = expr.data_type();
    let target_scale = expr.get_scale();
    let target_steal = expr.steal_sign();

    let mut out_bits = match target_dtype {
        DataType::Float(bits) => bits,
        DataType::Int(bits) => bits,
        DataType::DynamicFloat => 64,
        DataType::FloatingInt => 64,
        DataType::ScalableInt(bits) => bits,
        DataType::ScalableFloat(bits, _) => bits,
    };

    let is_float_mode = !matches!(
        target_dtype,
        DataType::Int(_) | DataType::ScalableInt(_) | DataType::FloatingInt
    );

    let mut original_bit_widths = HashMap::new();
    collect_original_widths(expr, &mut original_bit_widths);
    let compute_bits = override_compute_bits.unwrap_or(out_bits);
    let run_fn = compile_packed_kernel(&kernel, is_float_mode, compute_bits, out_bits, &original_bit_widths)
        .expect("Failed to compile Secondary JIT kernel");

    let input_ptrs: Vec<*const u8> = kernel.inputs.iter().map(|input_var| {
        match input_var {
            Expr::Variable { packed_data, .. } => packed_data.as_ptr(),
            _ => unreachable!(),
        }
    }).collect();

    let out_bytes = if out_bits % 8 == 0 {
        kernel.size * (out_bits as usize / 8)
    } else {
        (kernel.size * out_bits as usize + 7) / 8
    };
    let mut output_buf = vec![0u8; out_bytes];

    unsafe {
        run_fn(input_ptrs.as_ptr(), output_buf.as_mut_ptr(), kernel.size);
    }

    let mut final_dtype = target_dtype;
    let mut final_packed = output_buf;

    if matches!(target_dtype, DataType::ScalableInt(_) | DataType::ScalableFloat(_, _)) {
        let raw = engine.unpack(&final_packed, kernel.size, out_bits);
        let k = ((out_bits + 63) / 64) as usize;
        let output_vals: Vec<f64> = raw.chunks_exact(k)
            .map(|limbs| Scalar::from_limbs(limbs, target_dtype, target_scale, target_steal).to_double())
            .collect();
        
        match target_dtype {
            DataType::ScalableInt(_) => {
                let max_bits = output_vals.iter()
                    .map(|&v| crate::types::compute_scalable_int_bits(v.round() as i64))
                    .max()
                    .unwrap_or(2);
                final_dtype = DataType::ScalableInt(max_bits);
                if max_bits != out_bits {
                    let output_u64s: Vec<u64> = output_vals.iter().map(|&f| {
                        Scalar::ScalableInt(f.round() as i64, max_bits).to_u64_packed(target_steal)
                    }).collect();
                    final_packed = engine.pack(&output_u64s, max_bits);
                }
            }
            DataType::ScalableFloat(_, _) => {
                let mut max_b = 8;
                let mut max_e = 4;
                let sample_steal = target_steal && output_vals.iter().all(|&v| v >= 0.0);
                for &v in &output_vals {
                    let (b, e) = crate::types::compute_scalable_float_config(v, sample_steal);
                    if b > max_b { max_b = b; }
                    if e > max_e { max_e = e; }
                }
                let sign_bit = if sample_steal { 0 } else { 1 };
                if max_b < max_e + sign_bit + 1 {
                    max_b = max_e + sign_bit + 1;
                }
                if max_b > 64 {
                    max_b = 64;
                }
                final_dtype = DataType::ScalableFloat(max_b, max_e);
                if max_b != out_bits {
                    let output_u64s: Vec<u64> = output_vals.iter().map(|&f| {
                        Scalar::ScalableFloat(f, max_b, max_e).to_u64_packed(sample_steal)
                    }).collect();
                    final_packed = engine.pack(&output_u64s, max_b);
                }
            }
            _ => {}
        }
    }

    Expr::new_var(
        999,
        "fused_secondary_result",
        final_dtype,
        kernel.size,
        final_packed,
        target_scale,
        target_steal,
    )
}


pub fn execute_expr_tertiary(engine: &HardwareEngine, expr: &Expr) -> Expr {
    let mut min_bits_final = None;
    let mut is_optimized = false;

    let target_dtype = expr.data_type();
    let target_scale = expr.get_scale();
    let target_steal = expr.steal_sign();

    let out_bits = match target_dtype {
        DataType::Float(bits) => bits,
        DataType::Int(bits) => bits,
        DataType::DynamicFloat => 64,
        DataType::FloatingInt => 64,
        DataType::ScalableInt(bits) => bits,
        DataType::ScalableFloat(bits, _) => bits,
    };

    let is_float_mode = !matches!(
        target_dtype,
        DataType::Int(_) | DataType::ScalableInt(_) | DataType::FloatingInt
    );

    let kernel = MicroKernel::generate(expr);
    let mut original_bit_widths = HashMap::new();
    collect_original_widths(expr, &mut original_bit_widths);

    for iter in 0..3 {
        if is_optimized {
            break;
        }

        println!("[Tertiary Fuser] Iteration {}, target_dtype={:?}, out_bits={}", iter, target_dtype, out_bits);

        let compute_bits = out_bits;
        let run_fn = match compile_packed_kernel(&kernel, is_float_mode, compute_bits, out_bits, &original_bit_widths) {
            Some(f) => f,
            None => {
                return execute_expr_cpu(engine, expr);
            }
        };

        let sample_size = kernel.size.min(100);
        let sample_out_bytes = if out_bits % 8 == 0 {
            sample_size * (out_bits as usize / 8)
        } else {
            (sample_size * out_bits as usize + 7) / 8
        };
        let mut sample_out = vec![0u8; sample_out_bytes];

        let input_ptrs: Vec<*const u8> = kernel.inputs.iter().map(|input_var| {
            match input_var {
                Expr::Variable { packed_data, .. } => packed_data.as_ptr(),
                _ => std::ptr::null(),
            }
        }).collect();

        unsafe {
            run_fn(input_ptrs.as_ptr(), sample_out.as_mut_ptr(), sample_size);
        }

        let raw = engine.unpack(&sample_out, sample_size, out_bits);
        let k = ((out_bits + 63) / 64) as usize;
        let sample_vals: Vec<f64> = raw.chunks_exact(k)
            .map(|limbs| Scalar::from_limbs(limbs, target_dtype, target_scale, target_steal).to_double())
            .collect();

        println!("[Tertiary Fuser] Sample Vals: {:?}", sample_vals);

        let mut sample_steal = target_steal && sample_vals.iter().all(|&v| v >= 0.0);
        let mut min_bits = out_bits;

        match target_dtype {
            DataType::ScalableInt(_) | DataType::Int(_) => {
                let max_bits = sample_vals.iter()
                    .map(|&v| crate::types::compute_scalable_int_bits(v.round() as i64))
                    .max()
                    .unwrap_or(2);
                min_bits = max_bits;
            }
            DataType::ScalableFloat(_, _) | DataType::Float(_) => {
                let mut max_b = 8;
                let mut max_e = 4;
                for &v in &sample_vals {
                    let (b, e) = crate::types::compute_scalable_float_config(v, sample_steal);
                    if b > max_b { max_b = b; }
                    if e > max_e { max_e = e; }
                }
                let sign_bit = if sample_steal { 0 } else { 1 };
                if max_b < max_e + sign_bit + 1 {
                    max_b = max_e + sign_bit + 1;
                }
                if max_b > 64 {
                    max_b = 64;
                }
                min_bits = max_b;
            }
            _ => {}
        }

        if min_bits > out_bits {
            min_bits = out_bits;
        }

        println!("[Tertiary Fuser] min_bits calculated: {}", min_bits);

        if min_bits < out_bits || sample_steal != target_steal {
            println!("[Tertiary Fuser] Optimizing to min_bits={}", min_bits);
            min_bits_final = Some(min_bits);
            is_optimized = true; // Final iteration using optimized bits
        } else {
            println!("[Tertiary Fuser] Expression is fully optimized.");
            is_optimized = true;
        }
    }

    execute_expr_secondary_internal(engine, expr, min_bits_final)
}

// ─────────────────────────────────────────────────────────
// Unified dispatch: CPU or GPU
// ─────────────────────────────────────────────────────────

pub fn execute_expr(engine: &HardwareEngine, expr: &Expr) -> Expr {
    execute_expr_on_device(engine, expr, "cpu")
}

pub fn execute_expr_on_device(engine: &HardwareEngine, expr: &Expr, device: &str) -> Expr {
    match device {
        "gpu" => {
            execute_expr_gpu(engine, expr)
                .unwrap_or_else(|| {
                    eprintln!("[NumToy] GPU unavailable – falling back to CPU JIT");
                    execute_expr_cpu(engine, expr)
                })
        }
        "secondary" => {
            execute_expr_secondary(engine, expr)
        }
        "tertiary" => {
            execute_expr_tertiary(engine, expr)
        }
        _ => execute_expr_cpu(engine, expr),
    }
}

pub fn execute_expr_cpu(engine: &HardwareEngine, expr: &Expr) -> Expr {
    if needs_limb_interpreter(expr) {
        return execute_expr_limb_interpreter(engine, expr);
    }
    let kernel = MicroKernel::generate(expr);
    let steal_sign = expr.steal_sign();

    // 1. Unpack inputs into f64 buffers
    let mut input_bufs = Vec::new();
    for input_var in &kernel.inputs {
        match input_var {
            Expr::Variable { dtype, size, packed_data, scale, steal_sign: var_ss, .. } => {
                let bit_width = match dtype {
                    DataType::Float(bits) => *bits,
                    DataType::Int(bits) => *bits,
                    DataType::DynamicFloat => 64,
                    DataType::FloatingInt => 64,
                    DataType::ScalableInt(bits) => *bits,
                    DataType::ScalableFloat(bits, _) => *bits,
                };
                let raw_u64s = engine.unpack(packed_data, *size, bit_width);
                let f64s: Vec<f64> = raw_u64s
                    .into_iter()
                    .map(|u| Scalar::from_u64_packed(u, *dtype, *scale, *var_ss).to_double())
                    .collect();
                input_bufs.push(f64s);
            }
            _ => unreachable!(),
        }
    }

    // 2. JIT Compile & Run
    let run_fn = compile_kernel(&kernel).expect("Failed to compile JIT kernel");
    let input_ptrs: Vec<*const f64> = input_bufs.iter().map(|buf| buf.as_ptr()).collect();
    let mut output_buf = vec![0.0f64; kernel.size];

    run_fn(input_ptrs.as_ptr(), output_buf.as_mut_ptr(), kernel.size);

    // 3. Pack output back to target DataType
    let target_dtype = expr.data_type();
    let target_scale = expr.get_scale();
    let target_bit_width = match target_dtype {
        DataType::Float(bits) => bits,
        DataType::Int(bits) => bits,
        DataType::DynamicFloat => 64,
        DataType::FloatingInt => 64,
        DataType::ScalableInt(bits) => bits,
        DataType::ScalableFloat(bits, _) => bits,
    };

    // Determine output sign stealing (all results non-negative and parent stole sign)
    let output_steal = steal_sign && output_buf.iter().all(|&v| v >= 0.0);

    let output_u64s: Vec<u64> = output_buf
        .into_iter()
        .map(|f| {
            let scalar = match target_dtype {
                DataType::Float(bits) => Scalar::Float(f, bits),
                DataType::Int(bits) => Scalar::Int(f.round() as i64, bits),
                DataType::DynamicFloat => Scalar::DynamicFloat(f),
                DataType::FloatingInt => double_to_floating_int(f, target_scale.unwrap_or(1)),
                DataType::ScalableInt(bits) => Scalar::ScalableInt(f.round() as i64, bits),
                DataType::ScalableFloat(bits, e) => Scalar::ScalableFloat(f, bits, e),
            };
            scalar.to_u64_packed(output_steal)
        })
        .collect();

    let packed_output = engine.pack(&output_u64s, target_bit_width);

    Expr::new_var(
        999,
        "fused_result",
        target_dtype,
        kernel.size,
        packed_output,
        target_scale,
        output_steal,
    )
}

fn needs_limb_interpreter(expr: &Expr) -> bool {
    match expr {
        Expr::Variable { dtype, .. } => {
            match *dtype {
                DataType::Float(b) | DataType::Int(b) => b > 64,
                DataType::ScalableInt(_) | DataType::ScalableFloat(_, _) => true,
                _ => false,
            }
        }
        Expr::Constant { val } => {
            match val.data_type() {
                DataType::Float(b) | DataType::Int(b) => b > 64,
                DataType::ScalableInt(_) | DataType::ScalableFloat(_, _) => true,
                _ => false,
            }
        }
        Expr::Add { left, right } |
        Expr::Sub { left, right } |
        Expr::Mul { left, right } |
        Expr::Div { left, right } => {
            needs_limb_interpreter(left) || needs_limb_interpreter(right)
        }
    }
}

pub fn add_limbs(a: &[u64], b: &[u64], bits: u32) -> Vec<u64> {
    let k = ((bits + 63) / 64) as usize;
    let mut res = vec![0u64; k];
    let mut carry = 0u64;
    for i in 0..k {
        let va = a.get(i).copied().unwrap_or(0);
        let vb = b.get(i).copied().unwrap_or(0);
        let (sum1, c1) = va.overflowing_add(vb);
        let (sum2, c2) = sum1.overflowing_add(carry);
        res[i] = sum2;
        carry = if c1 || c2 { 1 } else { 0 };
    }
    let last_limb_bits = bits % 64;
    if last_limb_bits > 0 && k > 0 {
        let mask = (1u64 << last_limb_bits) - 1;
        res[k - 1] &= mask;
    }
    res
}

pub fn sub_limbs(a: &[u64], b: &[u64], bits: u32) -> Vec<u64> {
    let k = ((bits + 63) / 64) as usize;
    let mut res = vec![0u64; k];
    let mut borrow = 0u64;
    for i in 0..k {
        let va = a.get(i).copied().unwrap_or(0);
        let vb = b.get(i).copied().unwrap_or(0);
        let (diff1, b1) = va.overflowing_sub(vb);
        let (diff2, b2) = diff1.overflowing_sub(borrow);
        res[i] = diff2;
        borrow = if b1 || b2 { 1 } else { 0 };
    }
    let last_limb_bits = bits % 64;
    if last_limb_bits > 0 && k > 0 {
        let mask = (1u64 << last_limb_bits) - 1;
        res[k - 1] &= mask;
    }
    res
}

pub fn mul_limbs(a: &[u64], b: &[u64], bits: u32) -> Vec<u64> {
    let k = ((bits + 63) / 64) as usize;
    let mut res = vec![0u64; k];
    for i in 0..k {
        let va = a.get(i).copied().unwrap_or(0);
        if va == 0 { continue; }
        let mut carry = 0u128;
        for j in 0..(k - i) {
            let vb = b.get(j).copied().unwrap_or(0);
            let prod = (va as u128) * (vb as u128) + (res[i + j] as u128) + carry;
            res[i + j] = prod as u64;
            carry = prod >> 64;
        }
        let mut idx = i + (k - i);
        while carry > 0 && idx < k {
            let sum = (res[idx] as u128) + carry;
            res[idx] = sum as u64;
            carry = sum >> 64;
            idx += 1;
        }
    }
    let last_limb_bits = bits % 64;
    if last_limb_bits > 0 && k > 0 {
        let mask = (1u64 << last_limb_bits) - 1;
        res[k - 1] &= mask;
    }
    res
}

pub fn execute_expr_limb_interpreter(engine: &HardwareEngine, expr: &Expr) -> Expr {
    let mut unpacked_vars = std::collections::HashMap::new();
    fn collect_vars(node: &Expr, engine: &HardwareEngine, cache: &mut std::collections::HashMap<usize, (DataType, Vec<u64>, Option<u32>, bool)>) {
        match node {
            Expr::Variable { id, dtype, size, packed_data, scale, steal_sign, .. } => {
                if !cache.contains_key(id) {
                    let bit_width = match dtype {
                        DataType::Float(b) => *b,
                        DataType::Int(b) => *b,
                        DataType::DynamicFloat => 64,
                        DataType::FloatingInt => 64,
                        DataType::ScalableInt(b) => *b,
                        DataType::ScalableFloat(b, _) => *b,
                    };
                    let raw = engine.unpack(packed_data, *size, bit_width);
                    cache.insert(*id, (*dtype, raw, *scale, *steal_sign));
                }
            }
            Expr::Constant { .. } => {}
            Expr::Add { left, right } | Expr::Sub { left, right } | Expr::Mul { left, right } | Expr::Div { left, right } => {
                collect_vars(left, engine, cache);
                collect_vars(right, engine, cache);
            }
        }
    }
    collect_vars(expr, engine, &mut unpacked_vars);

    let size = expr.size();
    let target_dtype = expr.data_type();
    let target_scale = expr.get_scale();

    let mut output_values = vec![0.0f64; size];
    
    fn eval_node(
        node: &Expr,
        idx: usize,
        unpacked_vars: &std::collections::HashMap<usize, (DataType, Vec<u64>, Option<u32>, bool)>,
    ) -> Scalar {
        match node {
            Expr::Variable { id, dtype, scale, steal_sign, .. } => {
                let (var_dtype, raw, var_scale, var_steal) = unpacked_vars.get(id).unwrap();
                let bit_width = match var_dtype {
                    DataType::Float(b) => *b,
                    DataType::Int(b) => *b,
                    DataType::DynamicFloat => 64,
                    DataType::FloatingInt => 64,
                    DataType::ScalableInt(b) => *b,
                    DataType::ScalableFloat(b, _) => *b,
                };
                let k = ((bit_width + 63) / 64) as usize;
                let chunk = &raw[idx * k .. (idx + 1) * k];
                Scalar::from_limbs(chunk, *var_dtype, *var_scale, *var_steal)
            }
            Expr::Constant { val } => val.clone(),
            Expr::Add { left, right } => {
                let l = eval_node(left, idx, unpacked_vars);
                let r = eval_node(right, idx, unpacked_vars);
                apply_op(l, r, |a, b| a.wrapping_add(b), |a, b| a + b, "add")
            }
            Expr::Sub { left, right } => {
                let l = eval_node(left, idx, unpacked_vars);
                let r = eval_node(right, idx, unpacked_vars);
                apply_op(l, r, |a, b| a.wrapping_sub(b), |a, b| a - b, "sub")
            }
            Expr::Mul { left, right } => {
                let l = eval_node(left, idx, unpacked_vars);
                let r = eval_node(right, idx, unpacked_vars);
                apply_op(l, r, |a, b| a.wrapping_mul(b), |a, b| a * b, "mul")
            }
            Expr::Div { left, right } => {
                let l = eval_node(left, idx, unpacked_vars);
                let r = eval_node(right, idx, unpacked_vars);
                apply_op(l, r, |a, b| if b == 0 { 0 } else { a / b }, |a, b| if b == 0.0 { 0.0 } else { a / b }, "div")
            }
        }
    }

    fn apply_op<FI, FF>(l: Scalar, r: Scalar, op_int: FI, op_float: FF, op_name: &str) -> Scalar
    where
        FI: Fn(i64, i64) -> i64,
        FF: Fn(f64, f64) -> f64,
    {
        match (&l, &r) {
            (Scalar::Int(_, lb), Scalar::Int(_, rb)) |
            (Scalar::IntLimbs(_, lb), Scalar::IntLimbs(_, rb)) |
            (Scalar::Int(_, lb), Scalar::IntLimbs(_, rb)) |
            (Scalar::IntLimbs(_, lb), Scalar::Int(_, rb)) => {
                let max_b = *lb.max(rb);
                if max_b <= 64 {
                    let lv = match l { Scalar::Int(v, _) => v, _ => unreachable!() };
                    let rv = match r { Scalar::Int(v, _) => v, _ => unreachable!() };
                    Scalar::Int(op_int(lv, rv), max_b)
                } else {
                    let alimbs = l.to_extended_limbs(max_b);
                    let blimbs = r.to_extended_limbs(max_b);
                    let res_limbs = match op_name {
                        "add" => add_limbs(&alimbs, &blimbs, max_b),
                        "sub" => sub_limbs(&alimbs, &blimbs, max_b),
                        "mul" => mul_limbs(&alimbs, &blimbs, max_b),
                        _ => {
                            let lf = crate::types::decode_custom_int_limbs_to_double(&alimbs, max_b);
                            let rf = crate::types::decode_custom_int_limbs_to_double(&blimbs, max_b);
                            let res_val = if rf == 0.0 { 0.0 } else { lf / rf };
                            let final_limbs = crate::types::encode_custom_int_limbs_from_double(res_val, max_b);
                            return Scalar::IntLimbs(final_limbs, max_b);
                        }
                    };
                    Scalar::IntLimbs(res_limbs, max_b)
                }
            }
            (Scalar::ScalableInt(_, lb), Scalar::ScalableInt(_, rb)) |
            (Scalar::ScalableIntLimbs(_, lb), Scalar::ScalableIntLimbs(_, rb)) |
            (Scalar::ScalableInt(_, lb), Scalar::ScalableIntLimbs(_, rb)) |
            (Scalar::ScalableIntLimbs(_, lb), Scalar::ScalableInt(_, rb)) => {
                let max_b = *lb.max(rb);
                if max_b <= 64 {
                    let lv = match l { Scalar::ScalableInt(v, _) => v, _ => unreachable!() };
                    let rv = match r { Scalar::ScalableInt(v, _) => v, _ => unreachable!() };
                    Scalar::ScalableInt(op_int(lv, rv), max_b)
                } else {
                    let alimbs = l.to_extended_limbs(max_b);
                    let blimbs = r.to_extended_limbs(max_b);
                    let res_limbs = match op_name {
                        "add" => add_limbs(&alimbs, &blimbs, max_b),
                        "sub" => sub_limbs(&alimbs, &blimbs, max_b),
                        "mul" => mul_limbs(&alimbs, &blimbs, max_b),
                        _ => {
                            let lf = crate::types::decode_custom_int_limbs_to_double(&alimbs, max_b);
                            let rf = crate::types::decode_custom_int_limbs_to_double(&blimbs, max_b);
                            let res_val = if rf == 0.0 { 0.0 } else { lf / rf };
                            let final_limbs = crate::types::encode_custom_int_limbs_from_double(res_val, max_b);
                            return Scalar::ScalableIntLimbs(final_limbs, max_b);
                        }
                    };
                    Scalar::ScalableIntLimbs(res_limbs, max_b)
                }
            }
            _ => {
                let lf = l.to_double();
                let rf = r.to_double();
                let res_val = op_float(lf, rf);
                match l {
                    Scalar::Float(_, b) => Scalar::Float(res_val, b),
                    Scalar::ScalableFloat(_, b, e) => Scalar::ScalableFloat(res_val, b, e),
                    Scalar::DynamicFloat(_) => Scalar::DynamicFloat(res_val),
                    _ => Scalar::DynamicFloat(res_val),
                }
            }
        }
    }

    for idx in 0..size {
        let val = eval_node(expr, idx, &unpacked_vars);
        output_values[idx] = val.to_double();
    }

    let mut final_dtype = target_dtype;
    let mut output_steal = expr.steal_sign();

    match target_dtype {
        DataType::ScalableInt(_) => {
            let max_bits = output_values.iter()
                .map(|&v| crate::types::compute_scalable_int_bits(v.round() as i64))
                .max()
                .unwrap_or(2);
            final_dtype = DataType::ScalableInt(max_bits);
        }
        DataType::ScalableFloat(_, _) => {
            output_steal = output_steal && output_values.iter().all(|&v| v >= 0.0);
            let mut max_b = 8;
            let mut max_e = 4;
            for &v in &output_values {
                let (b, e) = crate::types::compute_scalable_float_config(v, output_steal);
                if b > max_b { max_b = b; }
                if e > max_e { max_e = e; }
            }
            let sign_bit = if output_steal { 0 } else { 1 };
            if max_b < max_e + sign_bit + 1 {
                max_b = max_e + sign_bit + 1;
            }
            if max_b > 64 {
                max_b = 64;
            }
            final_dtype = DataType::ScalableFloat(max_b, max_e);
        }
        DataType::Float(_) | DataType::DynamicFloat => {
            output_steal = output_steal && output_values.iter().all(|&v| v >= 0.0);
        }
        _ => {}
    }

    let final_bit_width = match final_dtype {
        DataType::Float(b) => b,
        DataType::Int(b) => b,
        DataType::DynamicFloat => 64,
        DataType::FloatingInt => 64,
        DataType::ScalableInt(b) => b,
        DataType::ScalableFloat(b, _) => b,
    };

    let output_u64s: Vec<u64> = output_values.iter().flat_map(|&f| {
        let scalar = match final_dtype {
            DataType::Float(b) => Scalar::Float(f, b),
            DataType::Int(b) => {
                if b <= 64 {
                    Scalar::Int(f.round() as i64, b)
                } else {
                    Scalar::IntLimbs(crate::types::encode_custom_int_limbs_from_double(f, b), b)
                }
            }
            DataType::DynamicFloat => Scalar::DynamicFloat(f),
            DataType::FloatingInt => double_to_floating_int(f, target_scale.unwrap_or(1)),
            DataType::ScalableInt(b) => {
                if b <= 64 {
                    Scalar::ScalableInt(f.round() as i64, b)
                } else {
                    Scalar::ScalableIntLimbs(crate::types::encode_custom_int_limbs_from_double(f, b), b)
                }
            }
            DataType::ScalableFloat(b, e) => Scalar::ScalableFloat(f, b, e),
        };
        scalar.to_limbs(output_steal)
    }).collect();

    let packed_output = engine.pack(&output_u64s, final_bit_width);

    Expr::new_var(
        999,
        "fused_limb_result",
        final_dtype,
        size,
        packed_output,
        target_scale,
        output_steal,
    )
}
