use std::collections::HashMap;
use cranelift_codegen::ir::{types as cl_types, AbiParam, InstBuilder, MemFlags, Type, Value};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use crate::graph::{ArenaGraph, Node, NodeId};
use crate::types::{DataType, Scalar};

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




pub enum OpResult {
    Float(Value),
    Int64(Value),
    Int128(Value, Value),
}

fn compile_arena_op(
    builder: &mut FunctionBuilder,
    graph: &ArenaGraph,
    node_id: NodeId,
    inputs_arg: Value,
    i_val: Value,
    ptr_type: Type,
    is_float_mode: bool,
    compute_bits: u32,
    local_funcs: &LocalFuncs,
    original_bit_widths: &HashMap<usize, u32>,
    input_indices: &HashMap<usize, usize>,
) -> OpResult {
    match &graph.nodes[node_id] {
        Node::Variable { id, dtype, scale, steal_sign, .. } => {
            let current_bit_width = match dtype {
                DataType::Float(b) | DataType::Int(b) => *b,
                DataType::DynamicFloat | DataType::FloatingInt => 64,
                DataType::ScalableInt(b) | DataType::ScalableFloat(b, _) => *b,
            };
            let original_bit_width = *original_bit_widths.get(id).unwrap_or(&current_bit_width);
            let e = match dtype {
                DataType::ScalableFloat(_, exp) => *exp,
                DataType::Float(_) => if current_bit_width <= 16 { 5 } else if current_bit_width <= 32 { 8 } else { 11 },
                _ => 0,
            };
            
            let idx = *input_indices.get(id).unwrap_or(&0);
            let ptr_size = if ptr_type == cl_types::I64 { 8 } else { 4 };
            let offset = (idx * ptr_size) as i32;
            let buf_ptr = builder.ins().load(ptr_type, MemFlags::new(), inputs_arg, offset);

            if is_float_mode {
                let k = ((original_bit_width + 63) / 64) as usize;
                let bit_width_val = builder.ins().iconst(cl_types::I32, original_bit_width as i64);
                let limb0_idx_val = builder.ins().iconst(cl_types::I32, 0);
                let call = builder.ins().call(local_funcs.load, &[buf_ptr, i_val, bit_width_val, limb0_idx_val]);
                let limb0 = builder.inst_results(call)[0];
                
                let limb1 = if k > 1 {
                    let limb1_idx_val = builder.ins().iconst(cl_types::I32, 1);
                    let call = builder.ins().call(local_funcs.load, &[buf_ptr, i_val, bit_width_val, limb1_idx_val]);
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
                let e_val = builder.ins().iconst(cl_types::I32, e as i64);
                let scale_val = builder.ins().iconst(cl_types::I32, scale.unwrap_or(0) as i64);
                let steal_val = builder.ins().iconst(cl_types::I8, if *steal_sign { 1 } else { 0 });

                let call_decode = builder.ins().call(
                    local_funcs.decode,
                    &[limb0, limb1, dtype_code_val, bit_width_val, e_val, scale_val, steal_val],
                );
                OpResult::Float(builder.inst_results(call_decode)[0])
            } else {
                let bit_width_val = builder.ins().iconst(cl_types::I32, original_bit_width as i64);
                let limb0_idx_val = builder.ins().iconst(cl_types::I32, 0);
                let call = builder.ins().call(local_funcs.load, &[buf_ptr, i_val, bit_width_val, limb0_idx_val]);
                let raw_low = builder.inst_results(call)[0];

                if compute_bits <= 64 {
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
                    let limb1_idx_val = builder.ins().iconst(cl_types::I32, 1);
                    let call1 = builder.ins().call(local_funcs.load, &[buf_ptr, i_val, bit_width_val, limb1_idx_val]);
                    let raw_high = builder.inst_results(call1)[0];
                    OpResult::Int128(raw_low, raw_high) // Real sign extension goes here if needed
                }
            }
        }
        Node::Constant { val } => {
            if is_float_mode {
                OpResult::Float(builder.ins().f64const(val.to_double()))
            } else {
                let v = val.to_double() as i64; // rough approx
                if compute_bits <= 64 {
                    OpResult::Int64(builder.ins().iconst(cl_types::I64, v))
                } else {
                    let low = builder.ins().iconst(cl_types::I64, v);
                    let high = builder.ins().iconst(cl_types::I64, if v < 0 { -1 } else { 0 });
                    OpResult::Int128(low, high)
                }
            }
        }
        Node::Add(left, right) | Node::Sub(left, right) | Node::Mul(left, right) | Node::Div(left, right) => {
            let l = compile_arena_op(builder, graph, *left, inputs_arg, i_val, ptr_type, is_float_mode, compute_bits, local_funcs, original_bit_widths, input_indices);
            let r = compile_arena_op(builder, graph, *right, inputs_arg, i_val, ptr_type, is_float_mode, compute_bits, local_funcs, original_bit_widths, input_indices);
            
            let is_add = matches!(&graph.nodes[node_id], Node::Add(..));
            let is_sub = matches!(&graph.nodes[node_id], Node::Sub(..));
            let is_mul = matches!(&graph.nodes[node_id], Node::Mul(..));
            let is_div = matches!(&graph.nodes[node_id], Node::Div(..));

            match (l, r) {
                (OpResult::Float(lf), OpResult::Float(rf)) => {
                    let res = if is_add { builder.ins().fadd(lf, rf) }
                        else if is_sub { builder.ins().fsub(lf, rf) }
                        else if is_mul { builder.ins().fmul(lf, rf) }
                        else { builder.ins().fdiv(lf, rf) };
                    OpResult::Float(res)
                }
                (OpResult::Int64(lv), OpResult::Int64(rv)) => {
                    let res = if is_add { builder.ins().iadd(lv, rv) }
                        else if is_sub { builder.ins().isub(lv, rv) }
                        else if is_mul { builder.ins().imul(lv, rv) }
                        else { builder.ins().sdiv(lv, rv) };
                    OpResult::Int64(res)
                }
                (OpResult::Int128(ll, lh), OpResult::Int128(rl, rh)) => {
                    let local_func = if is_add { local_funcs.add_i128 }
                        else if is_sub { local_funcs.sub_i128 }
                        else if is_mul { local_funcs.mul_i128 }
                        else { local_funcs.div_i128 };
                    
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

struct LocalFuncs {
    load: cranelift_codegen::ir::FuncRef,
    store: cranelift_codegen::ir::FuncRef,
    decode: cranelift_codegen::ir::FuncRef,
    encode: cranelift_codegen::ir::FuncRef,
    add_i128: cranelift_codegen::ir::FuncRef,
    sub_i128: cranelift_codegen::ir::FuncRef,
    mul_i128: cranelift_codegen::ir::FuncRef,
    div_i128: cranelift_codegen::ir::FuncRef,
}

pub fn compile_packed_kernel(graph: &ArenaGraph) -> Option<extern "C" fn(*const *const u8, *mut u8, usize)> {
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

    let mut sig_store = module.make_signature();
    sig_store.params.push(AbiParam::new(ptr_type));
    sig_store.params.push(AbiParam::new(ptr_type));
    sig_store.params.push(AbiParam::new(cl_types::I32));
    sig_store.params.push(AbiParam::new(cl_types::I32));
    sig_store.params.push(AbiParam::new(cl_types::I64));
    let func_store = module.declare_function("helper_store_u64", Linkage::Import, &sig_store).ok()?;

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

    let mut sig_bin128 = module.make_signature();
    sig_bin128.params.push(AbiParam::new(cl_types::I64));
    sig_bin128.params.push(AbiParam::new(cl_types::I64));
    sig_bin128.params.push(AbiParam::new(cl_types::I64));
    sig_bin128.params.push(AbiParam::new(cl_types::I64));
    sig_bin128.params.push(AbiParam::new(cl_types::I32));
    sig_bin128.returns.push(AbiParam::new(cl_types::I64));
    let func_add_i128 = module.declare_function("helper_add_i128", Linkage::Import, &sig_bin128).ok()?;
    let func_sub_i128 = module.declare_function("helper_sub_i128", Linkage::Import, &sig_bin128).ok()?;
    let func_mul_i128 = module.declare_function("helper_mul_i128", Linkage::Import, &sig_bin128).ok()?;
    let func_div_i128 = module.declare_function("helper_div_i128", Linkage::Import, &sig_bin128).ok()?;

    let local_funcs = LocalFuncs {
        load: module.declare_func_in_func(func_load, builder.func),
        store: module.declare_func_in_func(func_store, builder.func),
        decode: module.declare_func_in_func(func_decode, builder.func),
        encode: module.declare_func_in_func(func_encode, builder.func),
        add_i128: module.declare_func_in_func(func_add_i128, builder.func),
        sub_i128: module.declare_func_in_func(func_sub_i128, builder.func),
        mul_i128: module.declare_func_in_func(func_mul_i128, builder.func),
        div_i128: module.declare_func_in_func(func_div_i128, builder.func),
    };

    let target_dtype = graph.data_type(graph.root);
    let target_scale = graph.get_scale(graph.root);
    let target_steal = graph.steal_sign(graph.root);
    let is_float_mode = !matches!(
        target_dtype,
        DataType::Int(_) | DataType::ScalableInt(_) | DataType::FloatingInt
    );
    let out_bits = match target_dtype {
        DataType::Float(bits) | DataType::Int(bits) => bits,
        DataType::DynamicFloat | DataType::FloatingInt => 64,
        DataType::ScalableInt(bits) | DataType::ScalableFloat(bits, _) => bits,
    };
    let compute_bits = out_bits; // For now, fuser optimization overrides can change this.

    // Gather original_bit_widths and indices
    let mut original_bit_widths = HashMap::new();
    let mut input_indices = HashMap::new();
    let mut idx_counter = 0;
    for node in &graph.nodes {
        if let Node::Variable { id, dtype, .. } = node {
            let bw = match dtype {
                DataType::Float(bits) | DataType::Int(bits) => *bits,
                DataType::DynamicFloat | DataType::FloatingInt => 64,
                DataType::ScalableInt(bits) | DataType::ScalableFloat(bits, _) => *bits,
            };
            original_bit_widths.insert(*id, bw);
            if !input_indices.contains_key(id) {
                input_indices.insert(*id, idx_counter);
                idx_counter += 1;
            }
        }
    }

    // Unrolled Main Loop (stride 8)
    let unrolled_header = builder.create_block();
    let unrolled_body = builder.create_block();
    let cleanup_header = builder.create_block();
    let cleanup_body = builder.create_block();
    let loop_exit = builder.create_block();

    let idx_var = builder.declare_var(ptr_type);
    let zero = builder.ins().iconst(ptr_type, 0);
    builder.def_var(idx_var, zero);

    builder.ins().jump(unrolled_header, &[]);

    // Unrolled loop condition (i + 8 <= len)
    builder.switch_to_block(unrolled_header);
    let i_val = builder.use_var(idx_var);
    let eight = builder.ins().iconst(ptr_type, 8);
    let i_plus_8 = builder.ins().iadd(i_val, eight);
    let cmp = builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, i_plus_8, len_arg);
    builder.ins().brif(cmp, unrolled_body, &[], cleanup_header, &[]);

    // Unrolled loop body (execute 8 times)
    builder.switch_to_block(unrolled_body);
    for offset in 0..8 {
        let offset_val = builder.ins().iconst(ptr_type, offset as i64);
        let current_i = builder.ins().iadd(i_val, offset_val);

        let result_val = compile_arena_op(
            &mut builder,
            graph,
            graph.root,
            inputs_arg,
            current_i,
            ptr_type,
            is_float_mode,
            compute_bits,
            &local_funcs,
            &original_bit_widths,
            &input_indices,
        );

        store_result(
            &mut builder, result_val, current_i, output_arg, out_bits, is_float_mode, &local_funcs, target_dtype, target_scale, target_steal
        );
    }
    builder.def_var(idx_var, i_plus_8);
    builder.ins().jump(unrolled_header, &[]);

    // Cleanup loop condition (i < len)
    builder.switch_to_block(cleanup_header);
    let i_val_clean = builder.use_var(idx_var);
    let cmp_clean = builder.ins().icmp(IntCC::UnsignedLessThan, i_val_clean, len_arg);
    builder.ins().brif(cmp_clean, cleanup_body, &[], loop_exit, &[]);

    // Cleanup loop body (execute 1 time)
    builder.switch_to_block(cleanup_body);
    let result_val_clean = compile_arena_op(
        &mut builder,
        graph,
        graph.root,
        inputs_arg,
        i_val_clean,
        ptr_type,
        is_float_mode,
        compute_bits,
        &local_funcs,
        &original_bit_widths,
        &input_indices,
    );

    store_result(
        &mut builder, result_val_clean, i_val_clean, output_arg, out_bits, is_float_mode, &local_funcs, target_dtype, target_scale, target_steal
    );

    let one = builder.ins().iconst(ptr_type, 1);
    let next_i = builder.ins().iadd(i_val_clean, one);
    builder.def_var(idx_var, next_i);
    builder.ins().jump(cleanup_header, &[]);

    builder.seal_block(unrolled_header);
    builder.seal_block(unrolled_body);
    builder.seal_block(cleanup_header);
    builder.seal_block(cleanup_body);
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

fn store_result(
    builder: &mut FunctionBuilder,
    result_val: OpResult,
    i_val: Value,
    output_arg: Value,
    out_bits: u32,
    is_float_mode: bool,
    local_funcs: &LocalFuncs,
    target_dtype: DataType,
    target_scale: Option<u32>,
    target_steal: bool,
) {
    let out_k = ((out_bits + 63) / 64) as usize;
    if is_float_mode {
        let f_val = match result_val {
            OpResult::Float(v) => v,
            _ => unreachable!(),
        };
        for limb_idx in 0..out_k {
            let dtype_code = match target_dtype {
                DataType::Float(_) => 0,
                DataType::Int(_) => 1,
                DataType::DynamicFloat => 2,
                DataType::FloatingInt => 3,
                DataType::ScalableInt(_) => 4,
                DataType::ScalableFloat(_, _) => 5,
            };
            let e = match target_dtype {
                DataType::ScalableFloat(_, exp) => exp,
                DataType::Float(_) => if out_bits <= 16 { 5 } else if out_bits <= 32 { 8 } else { 11 },
                _ => 0,
            };
            let dtype_code_val = builder.ins().iconst(cl_types::I32, dtype_code);
            let bits_val = builder.ins().iconst(cl_types::I32, out_bits as i64);
            let e_val = builder.ins().iconst(cl_types::I32, e as i64); 
            let scale_val = builder.ins().iconst(cl_types::I32, target_scale.unwrap_or(0) as i64);
            let steal_val = builder.ins().iconst(cl_types::I8, if target_steal { 1 } else { 0 });
            let limb_idx_val = builder.ins().iconst(cl_types::I32, limb_idx as i64);
            let encoded_limb = builder.ins().call(
                local_funcs.encode,
                &[f_val, dtype_code_val, bits_val, e_val, scale_val, steal_val, limb_idx_val],
            );
            let encoded_limb_val = builder.inst_results(encoded_limb)[0];

            let out_bits_val = builder.ins().iconst(cl_types::I32, out_bits as i64);
            let out_limb_idx_val = builder.ins().iconst(cl_types::I32, limb_idx as i64);
            builder.ins().call(
                local_funcs.store,
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
                local_funcs.store,
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
                local_funcs.store,
                &[output_arg, i_val, out_bits_val, limb0_idx, limb0],
            );
            let limb1_idx = builder.ins().iconst(cl_types::I32, 1);
            builder.ins().call(
                local_funcs.store,
                &[output_arg, i_val, out_bits_val, limb1_idx, limb1],
            );
        }
    }
}

pub struct MicroKernel {
    pub size: usize,
    pub inputs: Vec<crate::ir::Expr>,
}

impl MicroKernel {
    pub fn generate(expr: &crate::ir::Expr) -> Self {
        let mut inputs = Vec::new();
        Self::collect_inputs(expr, &mut inputs);
        let size = expr.size();
        MicroKernel { size, inputs }
    }

    fn collect_inputs(expr: &crate::ir::Expr, inputs: &mut Vec<crate::ir::Expr>) {
        match expr {
            crate::ir::Expr::Variable { .. } => {
                inputs.push(expr.clone());
            }
            crate::ir::Expr::Constant { .. } => {}
            crate::ir::Expr::Add { left, right }
            | crate::ir::Expr::Sub { left, right }
            | crate::ir::Expr::Mul { left, right }
            | crate::ir::Expr::Div { left, right } => {
                Self::collect_inputs(left, inputs);
                Self::collect_inputs(right, inputs);
            }
        }
    }
}
