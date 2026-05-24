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
        Node::Variable { id: var_id, dtype, size, shape, bit_strides, .. } => {
            hasher.update(b"Var");
            
            // Include variable id so that different variables with the same shape
            // don't collide in the JIT cache.
            hasher.update(&var_id.to_le_bytes());

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

            // Hash shape and bit_strides so that transposed/sliced views
            // of the same data compile separate kernels.
            for &d in shape {
                hasher.update(&d.to_le_bytes());
            }
            for &s in bit_strides {
                hasher.update(&s.to_le_bytes());
            }
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
        Node::DequantizeMatmul(a, w, ..) => {
            hasher.update(b"DequantizeMatmul");
            hash_node(graph, *a, hasher);
            hash_node(graph, *w, hasher);
        }
    }
}
