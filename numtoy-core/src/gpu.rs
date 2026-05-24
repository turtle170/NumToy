use crate::graph::{ArenaGraph, Node, NodeId};
use crate::ir::Expr;
use crate::hardware::HardwareEngine;
use std::borrow::Cow;

pub fn execute_gpu(engine: &HardwareEngine, expr: &Expr, graph: &ArenaGraph) -> Expr {
    // Basic WebGPU Execution Pipeline for floating point values
    
    // We will unpack the variables on CPU first, pass them as f32 arrays to the GPU, 
    // run the math, and repack the result.
    
    let size = expr.size();
    if size == 0 {
        return expr.clone(); // Nothing to do
    }
    
    let result_vec = pollster::block_on(run_compute(engine, expr, graph, size));
    
    // Pack the f32 result back into the target dtype
    let target_dtype = expr.data_type();
    let target_scale = expr.get_scale();
    let target_steal = expr.steal_sign();
    
    let bit_width = match target_dtype {
        crate::types::DataType::Float(b) | crate::types::DataType::Int(b) => b,
        crate::types::DataType::ScalableInt(b) | crate::types::DataType::ScalableFloat(b, _) => b,
        _ => 64,
    };
    
    let scalars: Vec<crate::types::Scalar> = result_vec.iter().map(|&v| {
        match target_dtype {
            crate::types::DataType::Float(b)  => crate::types::Scalar::Float(v as f64, b),
            crate::types::DataType::Int(b)    => crate::types::Scalar::Int(v.round() as i64, b),
            crate::types::DataType::DynamicFloat => crate::types::Scalar::DynamicFloat(v as f64),
            crate::types::DataType::FloatingInt  => crate::types::double_to_floating_int(v as f64, target_scale.unwrap_or(1)),
            crate::types::DataType::ScalableInt(b) => crate::types::Scalar::ScalableInt(v.round() as i64, b),
            crate::types::DataType::ScalableFloat(b, e) => crate::types::Scalar::ScalableFloat(v as f64, b, e),
        }
    }).collect();
    
    let u64s: Vec<u64> = scalars.iter().flat_map(|s| s.to_limbs(target_steal)).collect();
    let packed = engine.pack(&u64s, bit_width);
    
    Expr::new_var(
        0,
        "gpu_res",
        target_dtype,
        size,
        crate::ir::PackedBuffer::Memory(std::sync::Arc::new(packed)),
        vec![size],
        crate::array::row_major_bit_strides(&[size], target_dtype.bit_width() as usize),
        0,
        target_scale,
        target_steal,
    )
}

async fn run_compute(engine: &HardwareEngine, expr: &Expr, graph: &ArenaGraph, size: usize) -> Vec<f32> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to find an appropriate adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .expect("Failed to create device");

    let mut bind_group_layout_entries = Vec::new();
    let mut bind_group_entries = Vec::new();
    let mut buffers = Vec::new();
    
    // We will extract all Variables from the graph and map them to bindings.
    // We will also generate the WGSL source.
    let mut wgsl_code = String::new();
    wgsl_code.push_str("
@group(0) @binding(0)
var<storage, read_write> output: array<f32>;\n");
    
    let mut input_count = 0;
    
    // A simple recursive closure to build the expression
    fn build_expr(id: NodeId, graph: &ArenaGraph, input_count: &mut u32) -> String {
        let node = graph.nodes[id].clone();
        match node {
            Node::Variable { .. } => {
                let current_idx = *input_count;
                *input_count += 1;
                format!("input_{}[global_id.x]", current_idx)
            }
            Node::Constant { val } => {
                format!("{:.5}", val.to_double() as f32)
            }
            Node::Add(l, r) => format!("({} + {})", build_expr(l, graph, input_count), build_expr(r, graph, input_count)),
            Node::Sub(l, r) => format!("({} - {})", build_expr(l, graph, input_count), build_expr(r, graph, input_count)),
            Node::Mul(l, r) => format!("({} * {})", build_expr(l, graph, input_count), build_expr(r, graph, input_count)),
            Node::Div(l, r) => format!("({} / {})", build_expr(l, graph, input_count), build_expr(r, graph, input_count)),
            Node::DequantizeMatmul(..) => "0.0".to_string()
        }
    }
    
    let mut input_vars = Vec::new();
    fn collect_vars<'a>(expr: &'a Expr, vars: &mut Vec<&'a Expr>) {
        match expr {
            Expr::Variable { .. } => vars.push(expr),
            Expr::Add { left, right } | Expr::Sub { left, right } | Expr::Mul { left, right } | Expr::Div { left, right } => {
                collect_vars(left, vars);
                collect_vars(right, vars);
            }
            Expr::DequantizeMatmul { .. } => {}
            _ => {}
        }
    }
    collect_vars(expr, &mut input_vars);
    
    for (i, var) in input_vars.iter().enumerate() {
        let binding_idx = (i + 1) as u32;
        wgsl_code.push_str(&format!("
@group(0) @binding({})
var<storage, read> input_{}: array<f32>;\n", binding_idx, i));
        
        let (packed_data, bits, dtype, scale, steal_sign) = match var {
            Expr::Variable { packed_data, dtype, scale, steal_sign, .. } => {
                (packed_data, dtype.bit_width(), *dtype, *scale, *steal_sign)
            }
            _ => unreachable!(),
        };
        
        let unpacked = engine.unpack(packed_data.as_slice(), size, bits);
        let k = ((bits + 63) / 64) as usize;
        let mut float_data: Vec<f32> = unpacked.chunks_exact(k)
            .map(|limbs| crate::types::Scalar::from_limbs(limbs, dtype, scale, steal_sign).to_double() as f32)
            .collect();
            
        // If broadcasting, fill the rest
        if float_data.len() == 1 && size > 1 {
            float_data = vec![float_data[0]; size];
        }
        
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Input Buffer {}", i)),
            size: (float_data.len() * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&float_data));
        
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: binding_idx,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        buffers.push(buffer);
    }
    
    // Create output buffer
    let output_buffer_size = (size * 4) as wgpu::BufferAddress;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    bind_group_layout_entries.insert(0, wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &bind_group_layout_entries,
    });
    
    bind_group_entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: output_buffer.as_entire_binding(),
    });
    for (i, buffer) in buffers.iter().enumerate() {
        bind_group_entries.push(wgpu::BindGroupEntry {
            binding: (i + 1) as u32,
            resource: buffer.as_entire_binding(),
        });
    }
    
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    let mut dummy_input_count = 0;
    let compute_expr = build_expr(graph.root, graph, &mut dummy_input_count);
    
    wgsl_code.push_str("
@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= ");
    wgsl_code.push_str(&size.to_string());
    wgsl_code.push_str("u) { return; }
    output[global_id.x] = ");
    wgsl_code.push_str(&compute_expr);
    wgsl_code.push_str(";\n}\n");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&wgsl_code)),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroup_count = ((size as u32) + 63) / 64;
        cpass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::Wait);
    
    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        result
    } else {
        panic!("failed to run compute on gpu!")
    }
}
