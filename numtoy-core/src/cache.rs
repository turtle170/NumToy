use std::sync::Arc;
use dashmap::DashMap;
use blake3::Hasher;
use once_cell::sync::Lazy;
use crate::graph::{ArenaGraph, Node, NodeId};
use crate::types::DataType;

pub type ExecFn = extern "C" fn(*const *const u8, *mut u8, usize);

pub static JIT_CACHE: Lazy<DashMap<[u8; 32], ExecFn>> = Lazy::new(|| DashMap::new());

pub fn hash_graph(graph: &ArenaGraph) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hash_node(graph, graph.root, &mut hasher);
    hasher.finalize().into()
}

fn hash_node(graph: &ArenaGraph, id: NodeId, hasher: &mut Hasher) {
    match &graph.nodes[id] {
        Node::Variable { dtype, size, .. } => {
            hasher.update(b"Var");
            
            let (tag, bits, extra) = match dtype {
                DataType::Float(b) => (0, *b, 0),
                DataType::Int(b) => (1, *b, 0),
                DataType::DynamicFloat => (2, 64, 0),
                DataType::FloatingInt => (3, 64, 0),
                DataType::ScalableInt(b) => (4, *b, 0),
                DataType::ScalableFloat(b, e) => (5, *b, *e),
            };
            hasher.update(&[tag]);
            hasher.update(&bits.to_le_bytes());
            hasher.update(&extra.to_le_bytes());

            hasher.update(&size.to_le_bytes());
        }
        Node::Constant { val } => {
            hasher.update(b"Const");
            let bytes = val.to_double().to_le_bytes();
            hasher.update(&bytes);
        }
        Node::Add(l, r) => {
            hasher.update(b"Add");
            hash_node(graph, *l, hasher);
            hash_node(graph, *r, hasher);
        }
        Node::Sub(l, r) => {
            hasher.update(b"Sub");
            hash_node(graph, *l, hasher);
            hash_node(graph, *r, hasher);
        }
        Node::Mul(l, r) => {
            hasher.update(b"Mul");
            hash_node(graph, *l, hasher);
            hash_node(graph, *r, hasher);
        }
        Node::Div(l, r) => {
            hasher.update(b"Div");
            hash_node(graph, *l, hasher);
            hash_node(graph, *r, hasher);
        }
    }
}
