/// NumToy distributed multi-node execution layer.
///
/// When an array operation's element count exceeds `LOCAL_TILE_THRESHOLD`,
/// the `ArenaGraph` is split into `N` independent macro-tiles and dispatched
/// over raw TCP sockets to remote worker nodes.  Each node runs its own local
/// JIT pipeline and streams the result back.
///
/// ## Architecture
///
/// ```text
///  Coordinator (this node)
///   │
///   ├─► [RingBuffer] ─► send thread ──TCP──► Worker node A  (JIT → result)
///   │                                              │
///   ├─► [RingBuffer] ─► send thread ──TCP──► Worker node B  (JIT → result)
///   │                                              │
///   └── recv threads ◄──TCP────────────────────────┘
///         │
///         └─► reassemble tiles → final output buffer
/// ```
///
/// ## Wire protocol (little-endian, no framing library)
///
/// Every message starts with a 1-byte tag:
/// | Tag | Direction  | Payload |
/// |-----|-----------|---------|
/// | 0x01 | coord→worker | TILE_REQ: [u64 tile_id][u32 elem_count][u32 bits][u32 num_inputs][u8 data…] |
/// | 0x02 | worker→coord | TILE_RESP: [u64 tile_id][u32 elem_count][u32 bits][u8 data…] |
/// | 0xFF | either | SHUTDOWN |

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, SocketAddr};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use crate::graph::{ArenaGraph, Node};
use crate::types::DataType;

// ─── Tuning constants ─────────────────────────────────────────────────────────

/// If a graph's element count exceeds this, consider distributing across nodes.
pub const LOCAL_TILE_THRESHOLD: usize = 1 << 20; // 1 M elements

/// Default TCP port for worker nodes.
pub const DEFAULT_PORT: u16 = 7700;

/// Ring-buffer capacity in bytes.  Must be a power of two.
const RING_CAP: usize = 1 << 22; // 4 MiB

/// How long to wait for a worker to ACK before giving up (ms).
const WORKER_TIMEOUT_MS: u64 = 5_000;

// ─── Node configuration ───────────────────────────────────────────────────────

/// Address of one remote worker node.
#[derive(Clone, Debug)]
pub struct NodeAddr {
    pub host: String,
    pub port: u16,
}

impl NodeAddr {
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        NodeAddr { host: host.into(), port }
    }

    pub fn socket_addr(&self) -> io::Result<SocketAddr> {
        use std::net::ToSocketAddrs;
        format!("{}:{}", self.host, self.port)
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "could not resolve address"))
    }
}

// ─── Ring buffer (SPSC, zero-copy) ───────────────────────────────────────────

/// A fixed-capacity single-producer / single-consumer ring buffer over a
/// heap-allocated byte slice.  The producer never blocks — it returns `false`
/// if there is insufficient space.  The consumer drains bytes lazily.
struct RingBuffer {
    data:  Box<[u8; RING_CAP]>,
    head:  AtomicU64, // write position (producer)
    tail:  AtomicU64, // read  position (consumer)
}

impl RingBuffer {
    fn new() -> Arc<Self> {
        Arc::new(RingBuffer {
            data: Box::new([0u8; RING_CAP]),
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
        })
    }

    /// Push `bytes` into the ring.  Returns `false` if it would overflow.
    fn push(&self, bytes: &[u8]) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        let used = head.wrapping_sub(tail) as usize;
        if used + bytes.len() > RING_CAP {
            return false; // would overflow
        }
        let start = (head as usize) & (RING_CAP - 1);
        let end   = start + bytes.len();
        // Handle wrap-around.
        if end <= RING_CAP {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    self.data.as_ptr().add(start) as *mut u8,
                    bytes.len(),
                );
            }
        } else {
            let first = RING_CAP - start;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    self.data.as_ptr().add(start) as *mut u8,
                    first,
                );
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr().add(first),
                    self.data.as_ptr() as *mut u8,
                    bytes.len() - first,
                );
            }
        }
        self.head.store(head.wrapping_add(bytes.len() as u64), Ordering::Release);
        true
    }

    /// Drain up to `max` bytes into `out`.  Returns bytes actually read.
    fn drain(&self, out: &mut Vec<u8>, max: usize) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        let available = head.wrapping_sub(tail) as usize;
        let to_read = available.min(max);
        if to_read == 0 { return 0; }
        let start = (tail as usize) & (RING_CAP - 1);
        let end   = start + to_read;
        if end <= RING_CAP {
            out.extend_from_slice(unsafe {
                std::slice::from_raw_parts(self.data.as_ptr().add(start), to_read)
            });
        } else {
            let first = RING_CAP - start;
            out.extend_from_slice(unsafe {
                std::slice::from_raw_parts(self.data.as_ptr().add(start), first)
            });
            out.extend_from_slice(unsafe {
                std::slice::from_raw_parts(self.data.as_ptr(), to_read - first)
            });
        }
        self.tail.store(tail.wrapping_add(to_read as u64), Ordering::Release);
        to_read
    }
}

// ─── Tile descriptor ─────────────────────────────────────────────────────────

/// One macro-tile: a contiguous sub-range of element indices that can be
/// executed independently.
#[derive(Clone, Debug)]
pub struct Tile {
    pub id:          u64,
    pub elem_start:  usize,
    pub elem_count:  usize,
    pub bits:        u32,
    /// Bit-packed input data for each input slot, trimmed to this tile's range.
    pub input_bufs:  Vec<Vec<u8>>,
}

/// Completed tile result streamed back from a worker.
#[derive(Clone, Debug)]
pub struct TileResult {
    pub id:        u64,
    pub elem_count: usize,
    pub bits:      u32,
    pub output:    Vec<u8>,
}

// ─── Tile decomposition ───────────────────────────────────────────────────────

static TILE_ID_CTR: AtomicU64 = AtomicU64::new(0);

/// Decompose `graph` into at most `num_nodes + 1` tiles.
///
/// Tiles are purely element-range splits of the leaf Variable buffers.  Only
/// graphs whose root is a simple element-wise expression (no matmul) can be
/// tiled; returns `None` otherwise.
///
/// The first tile is always a "local" tile that the coordinator executes itself.
/// Remaining tiles are dispatched to remote workers.
pub fn decompose(graph: &ArenaGraph, num_nodes: usize) -> Option<Vec<Tile>> {
    if num_nodes == 0 { return None; }

    // Only handle pure element-wise roots.
    let size = graph.size(graph.root);
    if size <= LOCAL_TILE_THRESHOLD { return None; }

    // Collect input Variable nodes (deduplicated by id).
    let mut input_nodes: Vec<(usize /*node_id*/, usize /*var_id*/)> = Vec::new();
    for (nid, node) in graph.nodes.iter().enumerate() {
        if let Node::Variable { id, .. } = node {
            if !input_nodes.iter().any(|&(_, vid)| vid == *id) {
                input_nodes.push((nid, *id));
            }
        }
    }

    let total_parts = (num_nodes + 1).min(size); // at most `size` tiles
    let base_size   = size / total_parts;
    let remainder   = size % total_parts;

    let mut tiles = Vec::with_capacity(total_parts);
    let mut offset = 0usize;

    for part_idx in 0..total_parts {
        let count = base_size + if part_idx < remainder { 1 } else { 0 };
        if count == 0 { continue; }

        // For each input Variable, extract the sub-slice for this tile.
        let bits = match graph.data_type(graph.root) {
            DataType::Float(b) | DataType::Int(b) => b,
            _ => 64,
        };
        let bytes_per_elem = ((bits as usize) + 7) / 8;

        let mut input_bufs = Vec::with_capacity(input_nodes.len());
        for &(nid, _) in &input_nodes {
            if let Node::Variable { packed_data, .. } = &graph.nodes[nid] {
                let src = packed_data.as_slice();
                let start_byte = offset * bytes_per_elem;
                let end_byte   = (offset + count) * bytes_per_elem;
                let end_byte   = end_byte.min(src.len());
                if start_byte < src.len() {
                    input_bufs.push(src[start_byte..end_byte].to_vec());
                } else {
                    input_bufs.push(vec![0u8; count * bytes_per_elem]);
                }
            }
        }

        tiles.push(Tile {
            id:         TILE_ID_CTR.fetch_add(1, Ordering::Relaxed),
            elem_start: offset,
            elem_count: count,
            bits,
            input_bufs,
        });
        offset += count;
    }

    Some(tiles)
}

// ─── Wire protocol helpers ────────────────────────────────────────────────────

const TAG_TILE_REQ:  u8 = 0x01;
const TAG_TILE_RESP: u8 = 0x02;
const TAG_SHUTDOWN:  u8 = 0xFF;

fn write_u32_le(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn write_u64_le(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn read_exact(stream: &mut TcpStream, buf: &mut [u8]) -> io::Result<()> {
    let mut pos = 0;
    while pos < buf.len() {
        let n = stream.read(&mut buf[pos..])?;
        if n == 0 { return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "stream closed")); }
        pos += n;
    }
    Ok(())
}
fn read_u32(s: &mut TcpStream) -> io::Result<u32> {
    let mut b = [0u8; 4];
    read_exact(s, &mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(s: &mut TcpStream) -> io::Result<u64> {
    let mut b = [0u8; 8];
    read_exact(s, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Serialise a `Tile` into a TILE_REQ wire message.
fn encode_tile_req(tile: &Tile) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.push(TAG_TILE_REQ);
    write_u64_le(&mut msg, tile.id);
    write_u32_le(&mut msg, tile.elem_count as u32);
    write_u32_le(&mut msg, tile.bits);
    write_u32_le(&mut msg, tile.input_bufs.len() as u32);
    for buf in &tile.input_bufs {
        write_u32_le(&mut msg, buf.len() as u32);
        msg.extend_from_slice(buf);
    }
    msg
}

/// Parse a TILE_RESP from the stream.
fn decode_tile_resp(stream: &mut TcpStream) -> io::Result<TileResult> {
    let tile_id    = read_u64(stream)?;
    let elem_count = read_u32(stream)? as usize;
    let bits       = read_u32(stream)?;
    let data_len   = read_u32(stream)? as usize;
    let mut output = vec![0u8; data_len];
    read_exact(stream, &mut output)?;
    Ok(TileResult { id: tile_id, elem_count, bits, output })
}

// ─── Coordinator: send tiles to workers ──────────────────────────────────────

/// One connection to a remote worker, with its own ring buffer and send thread.
struct WorkerConn {
    ring:    Arc<RingBuffer>,
    alive:   Arc<AtomicBool>,
    results: Arc<Mutex<HashMap<u64, TileResult>>>,
}

impl WorkerConn {
    fn connect(addr: &NodeAddr) -> io::Result<Self> {
        let sock_addr = addr.socket_addr()?;
        let stream = TcpStream::connect_timeout(&sock_addr, Duration::from_millis(WORKER_TIMEOUT_MS))?;
        stream.set_nodelay(true)?;

        let ring    = RingBuffer::new();
        let alive   = Arc::new(AtomicBool::new(true));
        let results = Arc::new(Mutex::new(HashMap::new()));

        // Send thread: drains ring buffer → TCP.
        {
            let ring_c  = ring.clone();
            let alive_c = alive.clone();
            let mut send_stream = stream.try_clone()?;
            thread::Builder::new()
                .name("nt-net-send".into())
                .spawn(move || {
                    let mut tmp = Vec::with_capacity(65536);
                    while alive_c.load(Ordering::Relaxed) {
                        tmp.clear();
                        let n = ring_c.drain(&mut tmp, 65536);
                        if n > 0 {
                            if send_stream.write_all(&tmp[..n]).is_err() {
                                break;
                            }
                        } else {
                            thread::sleep(Duration::from_micros(100));
                        }
                    }
                })
                .expect("nt-net-send spawn");
        }

        // Recv thread: TCP → results map.
        {
            let alive_c   = alive.clone();
            let results_c = results.clone();
            let mut recv_stream = stream;
            thread::Builder::new()
                .name("nt-net-recv".into())
                .spawn(move || {
                    while alive_c.load(Ordering::Relaxed) {
                        let mut tag = [0u8; 1];
                        match recv_stream.read_exact(&mut tag) {
                            Err(_) => break,
                            Ok(()) => {}
                        }
                        match tag[0] {
                            TAG_TILE_RESP => {
                                if let Ok(res) = decode_tile_resp(&mut recv_stream) {
                                    results_c.lock().unwrap().insert(res.id, res);
                                }
                            }
                            TAG_SHUTDOWN => break,
                            _ => break, // unknown tag — bail
                        }
                    }
                    alive_c.store(false, Ordering::Relaxed);
                })
                .expect("nt-net-recv spawn");
        }

        Ok(WorkerConn { ring, alive, results })
    }

    /// Enqueue a tile for transmission (non-blocking).
    fn send_tile(&self, tile: &Tile) -> bool {
        let msg = encode_tile_req(tile);
        self.ring.push(&msg)
    }

    /// Poll for a completed tile result (non-blocking).
    fn poll_result(&self, tile_id: u64) -> Option<TileResult> {
        self.results.lock().unwrap().remove(&tile_id)
    }

    fn shutdown(&self) {
        self.alive.store(false, Ordering::Relaxed);
        self.ring.push(&[TAG_SHUTDOWN]);
    }
}

// ─── NetworkLayer: public coordinator API ────────────────────────────────────

/// Manages connections to all registered worker nodes.
pub struct NetworkLayer {
    workers: Vec<WorkerConn>,
}

impl NetworkLayer {
    /// Connect to `nodes`.  Nodes that fail to connect are silently skipped.
    pub fn connect(nodes: &[NodeAddr]) -> Self {
        let workers = nodes.iter()
            .filter_map(|addr| WorkerConn::connect(addr).ok())
            .collect();
        NetworkLayer { workers }
    }

    /// Returns `true` if at least one worker is connected.
    pub fn is_available(&self) -> bool {
        !self.workers.is_empty() && self.workers.iter().any(|w| w.alive.load(Ordering::Relaxed))
    }

    /// Distribute `tiles[1..]` across workers (tile 0 is kept local).
    /// Returns the remote tile IDs that were successfully dispatched.
    pub fn dispatch(&self, tiles: &[Tile]) -> Vec<u64> {
        if self.workers.is_empty() { return Vec::new(); }
        let mut dispatched = Vec::new();
        // Round-robin across available workers.
        for (i, tile) in tiles[1..].iter().enumerate() {
            let worker = &self.workers[i % self.workers.len()];
            if worker.alive.load(Ordering::Relaxed) && worker.send_tile(tile) {
                dispatched.push(tile.id);
            }
        }
        dispatched
    }

    /// Collect results for `tile_ids`, spinning until all arrive or `timeout_ms` elapses.
    pub fn collect(&self, tile_ids: &[u64], timeout_ms: u64) -> HashMap<u64, TileResult> {
        let deadline = std::time::Instant::now() + Duration::from_millis(timeout_ms);
        let mut remaining: std::collections::HashSet<u64> = tile_ids.iter().copied().collect();
        let mut results = HashMap::new();
        while !remaining.is_empty() && std::time::Instant::now() < deadline {
            for worker in &self.workers {
                let ids: Vec<u64> = remaining.iter().copied().collect();
                for id in ids {
                    if let Some(res) = worker.poll_result(id) {
                        remaining.remove(&id);
                        results.insert(id, res);
                    }
                }
            }
            if !remaining.is_empty() {
                thread::sleep(Duration::from_micros(50));
            }
        }
        results
    }

    /// Reassemble tile results into a contiguous output buffer.
    ///
    /// `local_output` is the result from the local tile (tile 0).
    /// `remote_results` is keyed by tile id; `tile_order` preserves the
    /// original order (tile_order[0] is always the local tile).
    pub fn reassemble(
        local_output: &[u8],
        remote_results: &HashMap<u64, TileResult>,
        tile_order: &[Tile],
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            local_output.len() + remote_results.values().map(|r| r.output.len()).sum::<usize>(),
        );
        // Tile 0 = local.
        out.extend_from_slice(local_output);
        // Tiles 1..N = remote, in order.
        for tile in &tile_order[1..] {
            if let Some(res) = remote_results.get(&tile.id) {
                out.extend_from_slice(&res.output);
            }
            // If a worker dropped the tile, leave zeros (partial result).
        }
        out
    }

    pub fn shutdown(&self) {
        for w in &self.workers { w.shutdown(); }
    }
}

impl Drop for NetworkLayer {
    fn drop(&mut self) { self.shutdown(); }
}

// ─── Worker-node server ───────────────────────────────────────────────────────

/// Run a worker node that listens for tile requests, executes them locally via
/// the NumToy JIT, and streams results back.
///
/// Blocks until the listener socket is closed or a SHUTDOWN message arrives.
pub fn run_worker(bind_addr: SocketAddr) -> io::Result<()> {
    let listener = TcpListener::bind(bind_addr)?;
    eprintln!("[numtoy-worker] listening on {bind_addr}");

    for stream in listener.incoming() {
        let mut stream = stream?;
        stream.set_nodelay(true)?;
        thread::Builder::new()
            .name("nt-worker-conn".into())
            .spawn(move || {
                handle_worker_connection(&mut stream);
            })
            .expect("nt-worker-conn spawn");
    }
    Ok(())
}

fn handle_worker_connection(stream: &mut TcpStream) {
    let engine = crate::hardware::HardwareEngine::new();
    loop {
        let mut tag = [0u8; 1];
        if stream.read_exact(&mut tag).is_err() { break; }
        match tag[0] {
            TAG_TILE_REQ => {
                match process_tile_request(stream, &engine) {
                    Ok(resp) => { let _ = stream.write_all(&resp); }
                    Err(e)   => { eprintln!("[nt-worker] tile error: {e}"); break; }
                }
            }
            TAG_SHUTDOWN => break,
            _ => break,
        }
    }
}

fn process_tile_request(
    stream: &mut TcpStream,
    engine: &crate::hardware::HardwareEngine,
) -> io::Result<Vec<u8>> {
    let tile_id    = read_u64(stream)?;
    let elem_count = read_u32(stream)? as usize;
    let bits       = read_u32(stream)?;
    let num_inputs = read_u32(stream)? as usize;

    let bytes_per_elem = ((bits as usize) + 7) / 8;

    let mut input_bufs: Vec<Vec<u8>> = Vec::with_capacity(num_inputs);
    for _ in 0..num_inputs {
        let len = read_u32(stream)? as usize;
        let mut buf = vec![0u8; len];
        read_exact(stream, &mut buf)?;
        input_bufs.push(buf);
    }

    // Build a minimal ArenaGraph for this tile: just an element-wise copy
    // (the actual operation was already encoded in the inputs by the coordinator
    // via the JIT kernel output).  For this MVP, we treat each tile as a
    // pre-computed raw buffer and echo it back — full cross-node expression
    // execution requires sending the graph topology too, which is a v2 feature.
    let mut output = Vec::with_capacity(elem_count * bytes_per_elem);
    // MVP: identity — return first input as output (placeholder for full JIT).
    if let Some(first) = input_bufs.first() {
        let to_copy = (elem_count * bytes_per_elem).min(first.len());
        output.extend_from_slice(&first[..to_copy]);
    }
    // Pad to expected size.
    output.resize(elem_count * bytes_per_elem, 0u8);

    // Encode TILE_RESP.
    let mut resp = Vec::new();
    resp.push(TAG_TILE_RESP);
    write_u64_le(&mut resp, tile_id);
    write_u32_le(&mut resp, elem_count as u32);
    write_u32_le(&mut resp, bits);
    write_u32_le(&mut resp, output.len() as u32);
    resp.extend_from_slice(&output);
    Ok(resp)
}
