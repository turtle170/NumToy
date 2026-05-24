
// This goes in fuser.rs
pub extern "C" fn helper_dequantize_matmul(
    inputs: *const *const u8,
    out_ptr: *mut f64, // Output is evaluated as f64 in JIT mostly, wait, in fuser.rs result might be f64
    m: usize,
    k: usize,
    n: usize,
    w_bits: u32,
    w_scale: u32,
    w_steal: u8,
) {
    unsafe {
        let act_ptr = (*inputs.add(0)) as *const f64; // assuming activations are evaluated as f64? Wait, PyTensor uses f64 internally for all Float types in JIT? No, in ArenaGraph JIT float mode uses f64!
        let w_ptr = *inputs.add(1);
        let steal_sign = w_steal != 0;

        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f64;
                for i in 0..k {
                    let act_val = *act_ptr.add(row * k + i);
                    
                    let w_idx = i * n + col;
                    let bit_offset = w_idx * (w_bits as usize);
                    let byte_idx = bit_offset / 8;
                    let bit_in_byte = bit_offset % 8;
                    
                    let mut raw_val = 0u64;
                    let bytes_to_copy = std::cmp::min(8, (k * n * (w_bits as usize) + 7) / 8 - byte_idx);
                    if bytes_to_copy > 0 {
                        std::ptr::copy_nonoverlapping(
                            w_ptr.add(byte_idx),
                            &mut raw_val as *mut _ as *mut u8,
                            bytes_to_copy
                        );
                    }
                    
                    let mut w_val = (raw_val >> bit_in_byte) & ((1 << w_bits) - 1);
                    
                    // We can decode using helper_decode_to_f64
                    // Since weights dtype is ScalableInt or ScalableFloat, we know w_bits.
                    // But wait, what if it's Int? 
                    // To do it natively:
                    let is_signed = !steal_sign;
                    let mut w_f64 = w_val as f64;
                    if is_signed && (w_val & (1 << (w_bits - 1)) != 0) {
                        w_f64 -= (1 << w_bits) as f64;
                    }
                    
                    sum += act_val * w_f64;
                }
                *out_ptr.add(row * n + col) = sum;
            }
        }
    }
}
