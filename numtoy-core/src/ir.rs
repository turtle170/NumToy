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
    Sub {
        left: Arc<Expr>,
        right: Arc<Expr>,
    },
    Mul {
        left: Arc<Expr>,
        right: Arc<Expr>,
    },
    Div {
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

    pub fn sub(self, other: Expr) -> Self {
        Expr::Sub {
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

    pub fn div(self, other: Expr) -> Self {
        Expr::Div {
            left: Arc::new(self),
            right: Arc::new(other),
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Expr::Variable { dtype, .. } => *dtype,
            Expr::Constant { val } => val.data_type(),
            Expr::Add { left, .. } | Expr::Sub { left, .. }
            | Expr::Mul { left, .. } | Expr::Div { left, .. } => left.data_type(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Expr::Variable { size, .. } => *size,
            Expr::Constant { .. } => 1,
            Expr::Add { left, right } | Expr::Sub { left, right }
            | Expr::Mul { left, right } | Expr::Div { left, right } => {
                let l = left.size();
                let r = right.size();
                if l > r { l } else { r }
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
            Expr::Add { left, right } | Expr::Sub { left, right }
            | Expr::Mul { left, right } | Expr::Div { left, right } => {
                left.get_scale().or_else(|| right.get_scale())
            }
        }
    }

    pub fn steal_sign(&self) -> bool {
        match self {
            Expr::Variable { steal_sign, .. } => *steal_sign,
            Expr::Constant { .. } => false,
            Expr::Add { left, .. } | Expr::Sub { left, .. }
            | Expr::Mul { left, .. } | Expr::Div { left, .. } => left.steal_sign(),
        }
    }

    /// Reverse-mode symbolic auto-differentiation.
    ///
    /// Returns the symbolic gradient of `self` with respect to the variable
    /// whose `id == wrt_id`.  The returned expression can then be JIT-compiled
    /// and executed exactly like any other `Expr`.
    ///
    /// Rules:
    ///   d(Variable(id)) / d(id) = 1      (zero otherwise)
    ///   d(Constant)     / d(id) = 0
    ///   d(a + b)        / d(id) = da/dx + db/dx
    ///   d(a - b)        / d(id) = da/dx - db/dx
    ///   d(a * b)        / d(id) = a * db/dx + b * da/dx   (product rule)
    ///   d(a / b)        / d(id) = (da/dx * b - a * db/dx) / b²  (quotient rule)
    pub fn grad(&self, wrt_id: usize) -> Expr {
        match self {
            Expr::Variable { id, dtype, size, .. } => {
                if *id == wrt_id {
                    Expr::new_const(Scalar::Float(1.0, 32))
                } else {
                    Expr::new_const(Scalar::Float(0.0, 32))
                }
            }
            Expr::Constant { .. } => Expr::new_const(Scalar::Float(0.0, 32)),

            Expr::Add { left, right } => {
                let dl = left.grad(wrt_id);
                let dr = right.grad(wrt_id);
                dl.add(dr)
            }

            Expr::Sub { left, right } => {
                let dl = left.grad(wrt_id);
                let dr = right.grad(wrt_id);
                dl.sub(dr)
            }

            // Product rule: d(a*b)/dx = da/dx * b + a * db/dx
            Expr::Mul { left, right } => {
                let dl = left.grad(wrt_id);
                let dr = right.grad(wrt_id);
                let term1 = dl.mul((**right).clone());
                let term2 = (**left).clone().mul(dr);
                term1.add(term2)
            }

            // Quotient rule: d(a/b)/dx = (da/dx * b - a * db/dx) / b²
            Expr::Div { left, right } => {
                let dl = left.grad(wrt_id);
                let dr = right.grad(wrt_id);
                let b = (**right).clone();
                let b_sq = b.clone().mul(b.clone());
                let numer = dl.mul(b).sub((**left).clone().mul(dr));
                numer.div(b_sq)
            }
        }
    }
}
