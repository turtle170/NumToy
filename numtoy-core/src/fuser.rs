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
    Mul(Box<FusedOp>, Box<FusedOp>),
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
                Expr::Constant { val } => {
                    FusedOp::ConstVal(val.to_double())
                }
                Expr::Add { left, right } => {
                    let l = traverse(left, inputs, var_map);
                    let r = traverse(right, inputs, var_map);
                    FusedOp::Add(Box::new(l), Box::new(r))
                }
                Expr::Mul { left, right } => {
                    let l = traverse(left, inputs, var_map);
                    let r = traverse(right, inputs, var_map);
                    FusedOp::Mul(Box::new(l), Box::new(r))
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
            FusedOp::Mul(left, right) => {
                let l = compile_op(builder, left, inputs_arg, i_val, ptr_type);
                let r = compile_op(builder, right, inputs_arg, i_val, ptr_type);
                builder.ins().fmul(l, r)
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
            // WGSL requires the decimal point for float literals
            if v.fract() == 0.0 {
                format!("{:.1}", v)
            } else {
                format!("{}", v)
            }
        }
        FusedOp::Add(l, r) => format!("({} + {})", wgsl_expr(l), wgsl_expr(r)),
        FusedOp::Mul(l, r) => format!("({} * {})", wgsl_expr(l), wgsl_expr(r)),
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
                    // Graceful CPU fallback if no GPU is available
                    eprintln!("[NumToy] GPU unavailable – falling back to CPU JIT");
                    execute_expr_cpu(engine, expr)
                })
        }
        _ => execute_expr_cpu(engine, expr),
    }
}

pub fn execute_expr_cpu(engine: &HardwareEngine, expr: &Expr) -> Expr {
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
