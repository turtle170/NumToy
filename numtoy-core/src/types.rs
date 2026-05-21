#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum DataType {
    Float(u32),       // bit width
    Int(u32),         // bit width
    DynamicFloat,     // adaptable dynamic-bit float
    FloatingInt,      // fixed-point representation
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    Float(f64, u32),     // value, bit width
    Int(i64, u32),       // value, bit width
    DynamicFloat(f64),   // adaptable dynamic float
    FloatingInt { value: i64, scale: u32 }, // fixed-point: value * 10^-scale
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

pub fn encode_custom_float(val: f64, bits: u32, steal_sign: bool) -> u64 {
    if val == 0.0 {
        return 0;
    }
    
    let e = if bits <= 16 { 
        5 
    } else if bits <= 32 { 
        8 
    } else {
        11
    };
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

pub fn decode_custom_float(val: u64, bits: u32, steal_sign: bool) -> f64 {
    if val == 0 {
        return 0.0;
    }
    
    let e = if bits <= 16 { 
        5 
    } else if bits <= 32 { 
        8 
    } else {
        11
    };
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

impl Scalar {
    pub fn to_u64_packed(self, steal_sign: bool) -> u64 {
        match self {
            Scalar::Float(val, bits) => encode_custom_float(val, bits, steal_sign),
            Scalar::Int(val, _bits) => val as u64,
            Scalar::DynamicFloat(val) => encode_custom_float(val, 64, steal_sign),
            Scalar::FloatingInt { value, .. } => value as u64,
        }
    }

    pub fn from_u64_packed(val: u64, dtype: DataType, scale: Option<u32>, steal_sign: bool) -> Self {
        match dtype {
            DataType::Float(bits) => {
                let decoded = decode_custom_float(val, bits, steal_sign);
                Scalar::Float(decoded, bits)
            }
            DataType::Int(bits) => {
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
            }
            DataType::DynamicFloat => {
                let decoded = decode_custom_float(val, 64, steal_sign);
                Scalar::DynamicFloat(decoded)
            }
            DataType::FloatingInt => Scalar::FloatingInt {
                value: val as i64,
                scale: scale.unwrap_or(1),
            },
        }
    }

    pub fn to_double(self) -> f64 {
        match self {
            Scalar::Float(val, _) => val,
            Scalar::Int(val, _) => val as f64,
            Scalar::DynamicFloat(val) => val,
            Scalar::FloatingInt { value, scale } => {
                value as f64 / 10.0_f64.powi(scale as i32)
            }
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Scalar::Float(_, bits) => DataType::Float(*bits),
            Scalar::Int(_, bits) => DataType::Int(*bits),
            Scalar::DynamicFloat(_) => DataType::DynamicFloat,
            Scalar::FloatingInt { .. } => DataType::FloatingInt,
        }
    }
}

pub fn double_to_floating_int(val: f64, scale: u32) -> Scalar {
    let multiplier = 10.0_f64.powi(scale as i32);
    let int_val = (val * multiplier).round() as i64;
    Scalar::FloatingInt { value: int_val, scale }
}
