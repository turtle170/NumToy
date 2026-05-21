use crate::types::{DataType, Scalar};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Expr {
    Variable {
        id: usize,
        name: String,
        dtype: DataType,
        size: usize,
        packed_data: Vec<u8>,
        scale: Option<u32>, // Specifically for FloatingInt scale tracking
        steal_sign: bool,   // AdaptableFloat Sign Stealer: all values are non-negative
    },
    Constant {
        val: Scalar,
    },
    Add {
        left: Arc<Expr>,
        right: Arc<Expr>,
    },
    Mul {
        left: Arc<Expr>,
        right: Arc<Expr>,
    },
}

impl Expr {
    pub fn new_var(
        id: usize,
        name: &str,
        dtype: DataType,
        size: usize,
        packed_data: Vec<u8>,
        scale: Option<u32>,
        steal_sign: bool,
    ) -> Self {
        Expr::Variable {
            id,
            name: name.to_string(),
            dtype,
            size,
            packed_data,
            scale,
            steal_sign,
        }
    }

    pub fn new_const(val: Scalar) -> Self {
        Expr::Constant { val }
    }

    pub fn add(self, other: Expr) -> Self {
        Expr::Add {
            left: Arc::new(self),
            right: Arc::new(other),
        }
    }

    pub fn mul(self, other: Expr) -> Self {
        Expr::Mul {
            left: Arc::new(self),
            right: Arc::new(other),
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Expr::Variable { dtype, .. } => *dtype,
            Expr::Constant { val } => val.data_type(),
            Expr::Add { left, .. } => left.data_type(),
            Expr::Mul { left, .. } => left.data_type(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Expr::Variable { size, .. } => *size,
            Expr::Constant { .. } => 1,
            Expr::Add { left, right } => {
                let l_size = left.size();
                let r_size = right.size();
                if l_size > r_size { l_size } else { r_size }
            }
            Expr::Mul { left, right } => {
                let l_size = left.size();
                let r_size = right.size();
                if l_size > r_size { l_size } else { r_size }
            }
        }
    }

    pub fn get_scale(&self) -> Option<u32> {
        match self {
            Expr::Variable { scale, .. } => *scale,
            Expr::Constant { val } => match val {
                Scalar::FloatingInt { scale, .. } => Some(*scale),
                _ => None,
            },
            Expr::Add { left, right } => left.get_scale().or_else(|| right.get_scale()),
            Expr::Mul { left, right } => left.get_scale().or_else(|| right.get_scale()),
        }
    }

    pub fn steal_sign(&self) -> bool {
        match self {
            Expr::Variable { steal_sign, .. } => *steal_sign,
            Expr::Constant { .. } => false,
            Expr::Add { left, .. } => left.steal_sign(),
            Expr::Mul { left, .. } => left.steal_sign(),
        }
    }
}

