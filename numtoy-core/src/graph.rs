use crate::ir::Expr;
use crate::types::{DataType, Scalar};

pub type NodeId = usize;

#[derive(Debug, Clone)]
pub enum Node {
    Variable {
        id: usize,
        name: String,
        dtype: DataType,
        size: usize,
        packed_data: Vec<u8>,
        scale: Option<u32>,
        steal_sign: bool,
    },
    Constant {
        val: Scalar,
    },
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
}

#[derive(Debug, Clone)]
pub struct ArenaGraph {
    pub nodes: Vec<Node>,
    pub root: NodeId,
    pub is_stabilized: bool,
}

impl ArenaGraph {
    pub fn new() -> Self {
        ArenaGraph {
            nodes: Vec::new(),
            root: 0,
            is_stabilized: false,
        }
    }

    pub fn push(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    pub fn from_expr(expr: &Expr) -> Self {
        let mut graph = ArenaGraph::new();
        let root = Self::traverse(expr, &mut graph);
        graph.root = root;
        graph
    }

    fn traverse(expr: &Expr, graph: &mut ArenaGraph) -> NodeId {
        match expr {
            Expr::Variable { id, name, dtype, size, packed_data, scale, steal_sign } => {
                graph.push(Node::Variable {
                    id: *id,
                    name: name.clone(),
                    dtype: *dtype,
                    size: *size,
                    packed_data: packed_data.clone(),
                    scale: *scale,
                    steal_sign: *steal_sign,
                })
            }
            Expr::Constant { val } => {
                graph.push(Node::Constant { val: val.clone() })
            }
            Expr::Add { left, right } => {
                let l = Self::traverse(left, graph);
                let r = Self::traverse(right, graph);
                graph.push(Node::Add(l, r))
            }
            Expr::Sub { left, right } => {
                let l = Self::traverse(left, graph);
                let r = Self::traverse(right, graph);
                graph.push(Node::Sub(l, r))
            }
            Expr::Mul { left, right } => {
                let l = Self::traverse(left, graph);
                let r = Self::traverse(right, graph);
                graph.push(Node::Mul(l, r))
            }
            Expr::Div { left, right } => {
                let l = Self::traverse(left, graph);
                let r = Self::traverse(right, graph);
                graph.push(Node::Div(l, r))
            }
        }
    }

    pub fn data_type(&self, id: NodeId) -> DataType {
        match &self.nodes[id] {
            Node::Variable { dtype, .. } => *dtype,
            Node::Constant { val } => val.data_type(),
            Node::Add(l, _) | Node::Sub(l, _) | Node::Mul(l, _) | Node::Div(l, _) => self.data_type(*l),
        }
    }

    pub fn size(&self, id: NodeId) -> usize {
        match &self.nodes[id] {
            Node::Variable { size, .. } => *size,
            Node::Constant { .. } => 1,
            Node::Add(l, r) | Node::Sub(l, r) | Node::Mul(l, r) | Node::Div(l, r) => {
                let ls = self.size(*l);
                let rs = self.size(*r);
                if ls > rs { ls } else { rs }
            }
        }
    }

    pub fn get_scale(&self, id: NodeId) -> Option<u32> {
        match &self.nodes[id] {
            Node::Variable { scale, .. } => *scale,
            Node::Constant { val } => match val {
                Scalar::FloatingInt { scale, .. } => Some(*scale),
                _ => None,
            },
            Node::Add(l, r) | Node::Sub(l, r) | Node::Mul(l, r) | Node::Div(l, r) => {
                self.get_scale(*l).or_else(|| self.get_scale(*r))
            }
        }
    }

    pub fn steal_sign(&self, id: NodeId) -> bool {
        match &self.nodes[id] {
            Node::Variable { steal_sign, .. } => *steal_sign,
            Node::Constant { .. } => false,
            Node::Add(l, _) | Node::Sub(l, _) | Node::Mul(l, _) | Node::Div(l, _) => self.steal_sign(*l),
        }
    }
}
