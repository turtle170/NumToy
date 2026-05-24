#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum DataType {
    Float(u32),       // bit width
    Int(u32),         // bit width
    DynamicFloat,     // adaptable dynamic-bit float
    FloatingInt,      // fixed-point representation
    ScalableInt(u32), // scalable int with current active bit width
    ScalableFloat(u32, u32), // scalable float with current active bit width and exponent bit width
}

#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    Float(f64, u32),     // value, bit width
    Int(i64, u32),       // value, bit width
    DynamicFloat(f64),   // adaptable dynamic float
    FloatingInt { value: i64, scale: u32 }, // fixed-point: value * 10^-scale
    ScalableInt(i64, u32), // value, current active bit width
    ScalableFloat(f64, u32, u32), // value, current active bit width, current exponent bit width
    IntLimbs(Vec<u64>, u32),
    ScalableIntLimbs(Vec<u64>, u32),
}

fn scale_float(val: f64) -> (f64, i32) {
    let bits = val.to_bits();
    let exponent_raw = ((bits >> 52) & 0x7FF) as i32;
    let fraction_bits = bits & 0xFFFFFFFFFFFFF;
    
    if exponent_raw == 0 {
        if fraction_bits == 0 {
            (1.0, -1023)
        } else {
            let mut normalized_val = val;
            let mut shift = 0;
            while normalized_val < 1.0 {
                normalized_val *= 2.0;
                shift -= 1;
            }
            (normalized_val, -1023 + shift)
        }
    } else {
        let exponent = exponent_raw - 1023;
        let mantissa = f64::from_bits((0x3FF << 52) | fraction_bits);
        (mantissa, exponent)
    }
}

pub fn compute_scalable_int_bits(val: i64) -> u32 {
    let abs_val = val.unsigned_abs();
    if abs_val == 0 {
        2 // 1 magnitude bit + 1 sign bit
    } else {
        let mag_bits = 64 - abs_val.leading_zeros();
        let mag_bits = mag_bits.clamp(1, 63);
        mag_bits + 1
    }
}

pub fn compute_scalable_float_config(val: f64, steal_sign: bool) -> (u32, u32) {
    if val == 0.0 || val.is_nan() || val.is_infinite() {
        return (8, 4); // Default small config
    }
    
    let (_, exponent_raw) = scale_float(val.abs());
    
    // Find min exponent bits needed for exponent_raw
    let mut e = 2;
    while e < 15 {
        let bias = (1 << (e - 1)) - 1;
        let min_exp = 1 - bias;
        let max_exp = ((1 << e) - 2) - bias;
        if exponent_raw >= min_exp && exponent_raw <= max_exp {
            break;
        }
        e += 1;
    }
    
    let mag = exponent_raw.abs();
    let m = if mag <= 4 {
        8
    } else if mag <= 10 {
        16
    } else if mag <= 20 {
        24
    } else if mag <= 40 {
        32
    } else {
        40
    };
    
    let sign_bit = if steal_sign { 0 } else { 1 };
    let mut b = e + m + sign_bit;
    if b > 64 {
        b = 64;
    }
    if b < 8 {
        b = 8;
    }
    
    let available = if b > e + sign_bit { b - e - sign_bit } else { 0 };
    if available == 0 {
        b = e + sign_bit + 1;
    }
    
    (b, e)
}

pub fn encode_custom_float_with_e(val: f64, bits: u32, e: u32, steal_sign: bool) -> u64 {
    if val == 0.0 {
        return 0;
    }
    let bias = (1 << (e - 1)) - 1;
    let is_negative = val < 0.0;
    let abs_val = val.abs();
    
    let (mantissa, exponent_raw) = scale_float(abs_val);
    let exponent = exponent_raw + bias;
    
    let m_bits = if steal_sign { bits - e } else { bits - e - 1 };
    
    if exponent >= (1 << e) - 1 {
        let exp_part = ((1 << e) - 1) as u64;
        let sign_bit = if !steal_sign && is_negative { 1 } else { 0 };
        return (sign_bit << (bits - 1)) | (exp_part << m_bits);
    }
    if exponent <= 0 {
        return 0;
    }
    
    let fraction = mantissa - 1.0;
    let max_mantissa = (1u64 << m_bits) - 1;
    let mantissa_encoded = (fraction * (1u64 << m_bits) as f64).round() as u64;
    let mantissa_encoded = mantissa_encoded.min(max_mantissa);
    
    let sign_bit = if !steal_sign && is_negative { 1 } else { 0 };
    
    if steal_sign {
        ((exponent as u64) << m_bits) | mantissa_encoded
    } else {
        (sign_bit << (bits - 1)) | ((exponent as u64) << m_bits) | mantissa_encoded
    }
}

pub fn decode_custom_float_with_e(val: u64, bits: u32, e: u32, steal_sign: bool) -> f64 {
    if val == 0 {
        return 0.0;
    }
    let bias = (1 << (e - 1)) - 1;
    let m_bits = if steal_sign { bits - e } else { bits - e - 1 };
    
    let mantissa_mask = (1u64 << m_bits) - 1;
    let mantissa_bits = val & mantissa_mask;
    let exponent_bits = (val >> m_bits) & ((1u64 << e) - 1);
    
    let sign_bit = if steal_sign {
        0
    } else {
        (val >> (bits - 1)) & 1
    };
    
    if exponent_bits == ((1 << e) - 1) as u64 {
        return if sign_bit == 1 { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    
    let exponent = exponent_bits as i32 - bias;
    let fraction = mantissa_bits as f64 / (1u64 << m_bits) as f64;
    let mantissa = 1.0 + fraction;
    
    let result = mantissa * 2.0_f64.powi(exponent);
    if sign_bit == 1 {
        -result
    } else {
        result
    }
}

pub fn encode_custom_float(val: f64, bits: u32, steal_sign: bool) -> u64 {
    let e = if bits <= 16 { 
        5 
    } else if bits <= 32 { 
        8 
    } else {
        11
    };
    encode_custom_float_with_e(val, bits, e, steal_sign)
}

pub fn decode_custom_float(val: u64, bits: u32, steal_sign: bool) -> f64 {
    let e = if bits <= 16 { 
        5 
    } else if bits <= 32 { 
        8 
    } else {
        11
    };
    decode_custom_float_with_e(val, bits, e, steal_sign)
}

fn set_bits(limbs: &mut [u64], value_limbs: &[u64], bit_offset: u32, bit_len: u32) {
    for i in 0..bit_len {
        let src_limb = (i / 64) as usize;
        let src_bit = (i % 64) as u32;
        let bit = if src_limb < value_limbs.len() {
            (value_limbs[src_limb] >> src_bit) & 1
        } else {
            0
        };
        
        let dest_limb = ((bit_offset + i) / 64) as usize;
        let dest_bit = ((bit_offset + i) % 64) as u32;
        if dest_limb < limbs.len() {
            if bit == 1 {
                limbs[dest_limb] |= 1 << dest_bit;
            } else {
                limbs[dest_limb] &= !(1 << dest_bit);
            }
        }
    }
}

fn get_bits(limbs: &[u64], bit_offset: u32, bit_len: u32) -> Vec<u64> {
    let k = ((bit_len + 63) / 64) as usize;
    let mut res = vec![0u64; k];
    for i in 0..bit_len {
        let src_limb = ((bit_offset + i) / 64) as usize;
        let src_bit = ((bit_offset + i) % 64) as u32;
        let bit = if src_limb < limbs.len() {
            (limbs[src_limb] >> src_bit) & 1
        } else {
            0
        };
        
        let dest_limb = (i / 64) as usize;
        let dest_bit = (i % 64) as u32;
        if dest_limb < res.len() && bit == 1 {
            res[dest_limb] |= 1 << dest_bit;
        }
    }
    res
}

pub fn encode_custom_float_limbs(val: f64, bits: u32, e: u32, steal_sign: bool) -> Vec<u64> {
    let k = ((bits + 63) / 64) as usize;
    let mut limbs = vec![0u64; k];
    if val == 0.0 {
        return limbs;
    }
    
    let bias = (1i128 << (e - 1)) - 1;
    let is_negative = val < 0.0;
    let abs_val = val.abs();
    
    let (mantissa, exponent_raw) = scale_float(abs_val);
    let exponent = (exponent_raw as i128) + bias;
    
    let m_bits = if steal_sign { bits - e } else { bits - e - 1 };
    
    if exponent >= (1i128 << e) - 1 {
        let exp_all_ones = (1u64 << e.min(64)) - 1;
        let exp_limbs = vec![exp_all_ones; ((e + 63) / 64) as usize];
        set_bits(&mut limbs, &exp_limbs, m_bits, e);
        if !steal_sign && is_negative {
            set_bits(&mut limbs, &[1], bits - 1, 1);
        }
        return limbs;
    }
    if exponent <= 0 {
        return limbs;
    }
    
    let fraction = mantissa - 1.0;
    let mut mantissa_limbs = vec![0u64; ((m_bits + 63) / 64) as usize];
    if m_bits <= 52 {
        let val_m = (fraction * (1u64 << m_bits) as f64).round() as u64;
        let max_m = if m_bits == 64 { u64::MAX } else { (1u64 << m_bits) - 1 };
        let val_m = val_m.min(max_m);
        mantissa_limbs[0] = val_m;
    } else {
        let val_m52 = (fraction * (1u64 << 52) as f64).round() as u64;
        let shift = m_bits - 52;
        let shift_limbs = (shift / 64) as usize;
        let shift_bits = shift % 64;
        if shift_limbs < mantissa_limbs.len() {
            mantissa_limbs[shift_limbs] = val_m52 << shift_bits;
        }
        if shift_limbs + 1 < mantissa_limbs.len() && shift_bits > 0 {
            mantissa_limbs[shift_limbs + 1] = val_m52 >> (64 - shift_bits);
        }
    }
    
    set_bits(&mut limbs, &mantissa_limbs, 0, m_bits);
    
    let exp_u128 = exponent as u128;
    let exp_limbs = [exp_u128 as u64, (exp_u128 >> 64) as u64];
    set_bits(&mut limbs, &exp_limbs, m_bits, e);
    
    if !steal_sign && is_negative {
        set_bits(&mut limbs, &[1], bits - 1, 1);
    }
    
    limbs
}

pub fn decode_custom_float_limbs(limbs: &[u64], bits: u32, e: u32, steal_sign: bool) -> f64 {
    if limbs.iter().all(|&x| x == 0) {
        return 0.0;
    }
    
    let m_bits = if steal_sign { bits - e } else { bits - e - 1 };
    
    let exp_limbs = get_bits(limbs, m_bits, e);
    let mut exponent_bits: u128 = 0;
    for (i, &limb) in exp_limbs.iter().enumerate() {
        if i < 2 {
            exponent_bits |= (limb as u128) << (i * 64);
        }
    }
    
    let sign_bit = if steal_sign {
        0
    } else {
        let s_limbs = get_bits(limbs, bits - 1, 1);
        s_limbs[0] & 1
    };
    
    let max_exp = (1i128 << e) - 1;
    if exponent_bits as i128 == max_exp {
        return if sign_bit == 1 { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    
    let bias = (1i128 << (e - 1)) - 1;
    let exponent = exponent_bits as i128 - bias;
    
    let mantissa_limbs = get_bits(limbs, 0, m_bits);
    let fraction = if m_bits <= 52 {
        let mut mantissa_val = 0u64;
        if !mantissa_limbs.is_empty() {
            mantissa_val = mantissa_limbs[0];
        }
        mantissa_val as f64 / (1u64 << m_bits) as f64
    } else {
        let shift = m_bits - 52;
        let shift_limbs = (shift / 64) as usize;
        let shift_bits = shift % 64;
        
        let mut val_m52 = 0u64;
        if shift_limbs < mantissa_limbs.len() {
            val_m52 |= mantissa_limbs[shift_limbs] >> shift_bits;
        }
        if shift_limbs + 1 < mantissa_limbs.len() && shift_bits > 0 {
            val_m52 |= mantissa_limbs[shift_limbs + 1] << (64 - shift_bits);
        }
        val_m52 &= (1u64 << 52) - 1;
        val_m52 as f64 / (1u64 << 52) as f64
    };
    
    let mantissa = 1.0 + fraction;
    let result = if exponent > 1023 {
        f64::INFINITY
    } else if exponent < -1022 {
        mantissa * 2.0_f64.powi(-1022) * 2.0_f64.powi((exponent + 1022) as i32)
    } else {
        mantissa * 2.0_f64.powi(exponent as i32)
    };
    
    if sign_bit == 1 {
        -result
    } else {
        result
    }
}

pub fn encode_custom_int_limbs(val: i64, bits: u32) -> Vec<u64> {
    let k = ((bits + 63) / 64) as usize;
    let mut limbs = vec![0u64; k];
    
    if val >= 0 {
        let uval = val as u64;
        if k > 0 {
            limbs[0] = uval;
        }
        let last_limb_bits = bits % 64;
        if last_limb_bits > 0 && k > 0 {
            let mask = (1u64 << last_limb_bits) - 1;
            limbs[k - 1] &= mask;
        }
    } else {
        let abs_val = val.unsigned_abs();
        if k > 0 {
            limbs[0] = abs_val;
        }
        for i in 0..k {
            limbs[i] = !limbs[i];
        }
        let mut carry = 1u64;
        for i in 0..k {
            let (sum, c) = limbs[i].overflowing_add(carry);
            limbs[i] = sum;
            if c {
                carry = 1;
            } else {
                carry = 0;
                break;
            }
        }
        let last_limb_bits = bits % 64;
        if last_limb_bits > 0 && k > 0 {
            let mask = (1u64 << last_limb_bits) - 1;
            limbs[k - 1] &= mask;
        }
    }
    limbs
}

pub fn encode_custom_int_limbs_from_double(val: f64, bits: u32) -> Vec<u64> {
    let k = ((bits + 63) / 64) as usize;
    let mut limbs = vec![0u64; k];
    if val == 0.0 || val.is_nan() || val.is_infinite() {
        return limbs;
    }
    let is_negative = val < 0.0;
    let mut abs_val = val.abs();
    
    for i in 0..k {
        let limb_val = (abs_val % 18446744073709551616.0) as u64;
        limbs[i] = limb_val;
        abs_val = (abs_val / 18446744073709551616.0).floor();
    }
    
    if is_negative {
        for i in 0..k {
            limbs[i] = !limbs[i];
        }
        let mut carry = 1u64;
        for i in 0..k {
            let (sum, c) = limbs[i].overflowing_add(carry);
            limbs[i] = sum;
            if c {
                carry = 1;
            } else {
                carry = 0;
                break;
            }
        }
    }
    
    let last_limb_bits = bits % 64;
    if last_limb_bits > 0 && k > 0 {
        let mask = (1u64 << last_limb_bits) - 1;
        limbs[k - 1] &= mask;
    }
    limbs
}

pub fn decode_custom_int_limbs(limbs: &[u64], bits: u32) -> i64 {
    if limbs.is_empty() {
        return 0;
    }
    let sign_limb_idx = ((bits - 1) / 64) as usize;
    let sign_bit_idx = (bits - 1) % 64;
    let is_negative = if sign_limb_idx < limbs.len() {
        ((limbs[sign_limb_idx] >> sign_bit_idx) & 1) == 1
    } else {
        false
    };
    
    if !is_negative {
        limbs[0] as i64
    } else {
        let k = limbs.len();
        let mut temp = limbs.to_vec();
        for i in 0..k {
            temp[i] = !temp[i];
        }
        let mut carry = 1u64;
        for i in 0..k {
            let (sum, c) = temp[i].overflowing_add(carry);
            temp[i] = sum;
            if c {
                carry = 1;
            } else {
                carry = 0;
                break;
            }
        }
        let last_limb_bits = bits % 64;
        if last_limb_bits > 0 && k > 0 {
            let mask = (1u64 << last_limb_bits) - 1;
            temp[k - 1] &= mask;
        }
        
        let abs_val = temp[0] as i64;
        -abs_val
    }
}

pub fn extend_limbs(limbs: &[u64], from_bits: u32, to_bits: u32) -> Vec<u64> {
    let from_k = ((from_bits + 63) / 64) as usize;
    let to_k = ((to_bits + 63) / 64) as usize;
    let mut res = vec![0u64; to_k];
    for i in 0..from_k.min(to_k) {
        res[i] = limbs.get(i).copied().unwrap_or(0);
    }
    let sign_limb_idx = ((from_bits - 1) / 64) as usize;
    let sign_bit_idx = (from_bits - 1) % 64;
    let is_negative = if sign_limb_idx < limbs.len() {
        ((limbs[sign_limb_idx] >> sign_bit_idx) & 1) == 1
    } else {
        false
    };
    if is_negative {
        if from_k > 0 && from_k <= to_k {
            let last_copied_bit_idx = from_bits % 64;
            if last_copied_bit_idx > 0 {
                let mask = !((1u64 << last_copied_bit_idx) - 1);
                res[from_k - 1] |= mask;
            }
        }
        for i in from_k..to_k {
            res[i] = 0xFFFFFFFFFFFFFFFFu64;
        }
    }
    let last_limb_bits = to_bits % 64;
    if last_limb_bits > 0 && to_k > 0 {
        let mask = (1u64 << last_limb_bits) - 1;
        res[to_k - 1] &= mask;
    }
    res
}

pub fn decode_custom_int_limbs_to_double(limbs: &[u64], bits: u32) -> f64 {
    if limbs.is_empty() {
        return 0.0;
    }
    let sign_limb_idx = ((bits - 1) / 64) as usize;
    let sign_bit_idx = (bits - 1) % 64;
    let is_negative = if sign_limb_idx < limbs.len() {
        ((limbs[sign_limb_idx] >> sign_bit_idx) & 1) == 1
    } else {
        false
    };
    if !is_negative {
        let mut val = 0.0;
        for (i, &limb) in limbs.iter().enumerate() {
            val += (limb as f64) * 2.0_f64.powi(64 * i as i32);
        }
        val
    } else {
        let k = limbs.len();
        let mut temp = limbs.to_vec();
        for i in 0..k {
            temp[i] = !temp[i];
        }
        let mut carry = 1u64;
        for i in 0..k {
            let (sum, c) = temp[i].overflowing_add(carry);
            temp[i] = sum;
            if c {
                carry = 1;
            } else {
                carry = 0;
                break;
            }
        }
        let last_limb_bits = bits % 64;
        if last_limb_bits > 0 && k > 0 {
            let mask = (1u64 << last_limb_bits) - 1;
            temp[k - 1] &= mask;
        }
        let mut abs_val = 0.0;
        for (i, &limb) in temp.iter().enumerate() {
            abs_val += (limb as f64) * 2.0_f64.powi(64 * i as i32);
        }
        -abs_val
    }
}

impl Scalar {
    pub fn to_limbs(&self, steal_sign: bool) -> Vec<u64> {
        match self {
            Scalar::Float(val, bits) => {
                if *bits <= 64 {
                    vec![encode_custom_float(*val, *bits, steal_sign)]
                } else {
                    let e = if *bits <= 128 { 15 } else { 19 };
                    encode_custom_float_limbs(*val, *bits, e, steal_sign)
                }
            }
            Scalar::Int(val, bits) => {
                if *bits <= 64 {
                    vec![*val as u64]
                } else {
                    encode_custom_int_limbs(*val, *bits)
                }
            }
            Scalar::DynamicFloat(val) => {
                vec![encode_custom_float(*val, 64, steal_sign)]
            }
            Scalar::FloatingInt { value, .. } => {
                vec![*value as u64]
            }
            Scalar::ScalableInt(val, bits) => {
                if *bits <= 64 {
                    vec![*val as u64]
                } else {
                    encode_custom_int_limbs(*val, *bits)
                }
            }
            Scalar::ScalableFloat(val, bits, e) => {
                if *bits <= 64 {
                    vec![encode_custom_float_with_e(*val, *bits, *e, steal_sign)]
                } else {
                    encode_custom_float_limbs(*val, *bits, *e, steal_sign)
                }
            }
            Scalar::IntLimbs(limbs, _) => limbs.clone(),
            Scalar::ScalableIntLimbs(limbs, _) => limbs.clone(),
        }
    }

    pub fn to_extended_limbs(&self, max_b: u32) -> Vec<u64> {
        let original_bits = match self {
            Scalar::Int(_, b) => *b,
            Scalar::IntLimbs(_, b) => *b,
            Scalar::ScalableInt(_, b) => *b,
            Scalar::ScalableIntLimbs(_, b) => *b,
            _ => 64,
        };
        let limbs = self.to_limbs(false);
        extend_limbs(&limbs, original_bits, max_b)
    }

    pub fn to_u64_packed(&self, steal_sign: bool) -> u64 {
        self.to_limbs(steal_sign).first().copied().unwrap_or(0)
    }

    pub fn from_limbs(limbs: &[u64], dtype: DataType, scale: Option<u32>, steal_sign: bool) -> Self {
        match dtype {
            DataType::Float(bits) => {
                if bits <= 64 {
                    let val = limbs.first().copied().unwrap_or(0);
                    let decoded = decode_custom_float(val, bits, steal_sign);
                    Scalar::Float(decoded, bits)
                } else {
                    let e = if bits <= 128 { 15 } else { 19 };
                    let decoded = decode_custom_float_limbs(limbs, bits, e, steal_sign);
                    Scalar::Float(decoded, bits)
                }
            }
            DataType::Int(bits) => {
                if bits <= 64 {
                    let val = limbs.first().copied().unwrap_or(0);
                    if bits < 64 {
                        let mask = 1 << (bits - 1);
                        let val_masked = val & ((1 << bits) - 1);
                        if (val_masked & mask) != 0 {
                            let extended = val_masked | !((1 << bits) - 1);
                            Scalar::Int(extended as i64, bits)
                        } else {
                            Scalar::Int(val_masked as i64, bits)
                        }
                    } else {
                        Scalar::Int(val as i64, bits)
                    }
                } else {
                    Scalar::IntLimbs(limbs.to_vec(), bits)
                }
            }
            DataType::DynamicFloat => {
                let val = limbs.first().copied().unwrap_or(0);
                let decoded = decode_custom_float(val, 64, steal_sign);
                Scalar::DynamicFloat(decoded)
            }
            DataType::FloatingInt => {
                let val = limbs.first().copied().unwrap_or(0);
                Scalar::FloatingInt {
                    value: val as i64,
                    scale: scale.unwrap_or(1),
                }
            }
            DataType::ScalableInt(bits) => {
                if bits <= 64 {
                    let val = limbs.first().copied().unwrap_or(0);
                    let signed_val = if bits < 64 {
                        let mask = 1 << (bits - 1);
                        let val_masked = val & ((1 << bits) - 1);
                        if (val_masked & mask) != 0 {
                            (val_masked | !((1 << bits) - 1)) as i64
                        } else {
                            val_masked as i64
                        }
                    } else {
                        val as i64
                    };
                    Scalar::ScalableInt(signed_val, bits)
                } else {
                    Scalar::ScalableIntLimbs(limbs.to_vec(), bits)
                }
            }
            DataType::ScalableFloat(bits, e) => {
                if bits <= 64 {
                    let val = limbs.first().copied().unwrap_or(0);
                    let decoded = decode_custom_float_with_e(val, bits, e, steal_sign);
                    Scalar::ScalableFloat(decoded, bits, e)
                } else {
                    let decoded = decode_custom_float_limbs(limbs, bits, e, steal_sign);
                    Scalar::ScalableFloat(decoded, bits, e)
                }
            }
        }
    }

    pub fn from_u64_packed(val: u64, dtype: DataType, scale: Option<u32>, steal_sign: bool) -> Self {
        Self::from_limbs(&[val], dtype, scale, steal_sign)
    }

    pub fn to_double(&self) -> f64 {
        match self {
            Scalar::Float(val, _) => *val,
            Scalar::Int(val, _) => *val as f64,
            Scalar::DynamicFloat(val) => *val,
            Scalar::FloatingInt { value, scale } => {
                *value as f64 / 10.0_f64.powi(*scale as i32)
            }
            Scalar::ScalableInt(val, _) => *val as f64,
            Scalar::ScalableFloat(val, _, _) => *val,
            Scalar::IntLimbs(limbs, bits) => decode_custom_int_limbs_to_double(limbs, *bits),
            Scalar::ScalableIntLimbs(limbs, bits) => decode_custom_int_limbs_to_double(limbs, *bits),
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Scalar::Float(_, bits) => DataType::Float(*bits),
            Scalar::Int(_, bits) => DataType::Int(*bits),
            Scalar::DynamicFloat(_) => DataType::DynamicFloat,
            Scalar::FloatingInt { .. } => DataType::FloatingInt,
            Scalar::ScalableInt(_, bits) => DataType::ScalableInt(*bits),
            Scalar::ScalableFloat(_, bits, e) => DataType::ScalableFloat(*bits, *e),
            Scalar::IntLimbs(_, bits) => DataType::Int(*bits),
            Scalar::ScalableIntLimbs(_, bits) => DataType::ScalableInt(*bits),
        }
    }
}

pub fn double_to_floating_int(val: f64, scale: u32) -> Scalar {
    let multiplier = 10.0_f64.powi(scale as i32);
    let int_val = (val * multiplier).round() as i64;
    Scalar::FloatingInt { value: int_val, scale }
}

impl DataType {
    pub fn bit_width(&self) -> u32 {
        match self {
            DataType::Float(b) | DataType::Int(b) | DataType::ScalableInt(b) | DataType::ScalableFloat(b, _) => *b,
            DataType::DynamicFloat | DataType::FloatingInt => 64,
        }
    }
}
