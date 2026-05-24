use std::sync::{Arc, Mutex, Condvar};
use std::thread;
use crossbeam_deque::{Worker, Stealer, Injector};
use crate::graph::ArenaGraph;
use crate::cache::{JIT_CACHE, hash_graph, ExecFn};
use crate::fuser::compile_packed_kernel;
use std::sync::atomic::{AtomicBool, Ordering};

#[repr(align(64))]
#[derive(Clone)]
pub enum Task {
    Tile { graph: Arc<ArenaGraph>, hash: [u8; 32] },
    Fuse { graph: Arc<ArenaGraph>, hash: [u8; 32] },
    Assemble { graph: Arc<ArenaGraph>, hash: [u8; 32] },
}

use once_cell::sync::Lazy;

pub static GLOBAL_POOL: Lazy<ThreadPool> = Lazy::new(|| ThreadPool::new());

/// Set to `true` while execution mode is `Xtreme`, `false` otherwise.
///
/// A single `Ordering::Relaxed` bool load is cheaper in the idle hot-loop than
/// loading + matching the full `ExecutionMode` u8, so worker threads read this
/// flag directly rather than calling `get_execution_mode()` on every idle iter.
pub static XTREME_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Dedicated work queue for the single-core Xtreme worker.
///
/// In Xtreme mode `submit()` pushes here instead of the regular pool injector.
/// The Xtreme worker drains this queue in a hot-spin loop, running the full
/// `Tile → Assemble` pipeline inline with no sub-queuing.
static XTREME_INJECTOR: Lazy<Injector<Task>> = Lazy::new(Injector::new);

/// Guards single-spawn of the Xtreme worker thread.
/// Once flipped to `true` the thread lives for the process lifetime.
static XTREME_WORKER_STARTED: AtomicBool = AtomicBool::new(false);

pub struct ThreadPool {
    injector: Arc<Injector<Task>>,
    stealers: Vec<Stealer<Task>>,
    active: Arc<AtomicBool>,
}

impl ThreadPool {
    pub fn new() -> Self {
        let num_cores = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let num_tilers = (num_cores / 4).max(1);
        let num_fusers = (num_cores / 2).max(1);
        let num_assemblers = num_cores - num_tilers - num_fusers;
        
        let injector = Arc::new(Injector::new());
        let mut stealers = Vec::new();
        let mut workers = Vec::new();

        for _ in 0..num_cores {
            let worker = Worker::new_fifo();
            stealers.push(worker.stealer());
            workers.push(worker);
        }

        let active = Arc::new(AtomicBool::new(true));

        let core_ids = core_affinity::get_core_ids().unwrap_or_else(|| vec![]);

        for (i, worker) in workers.into_iter().enumerate() {
            let role = if i < num_tilers {
                "tiler"
            } else if i < num_tilers + num_fusers {
                "fuser"
            } else {
                "assembler"
            };

            let stealers_clone = stealers.clone();
            let injector_clone = injector.clone();
            let active_clone = active.clone();
            
            let core_id = core_ids.get(i % core_ids.len().max(1)).copied();

            thread::spawn(move || {
                if let Some(id) = core_id {
                    core_affinity::set_for_current(id);
                }
                worker_loop(worker, stealers_clone, injector_clone, active_clone, role);
            });
        }

        ThreadPool {
            injector,
            stealers,
            active,
        }
    }

    pub fn submit(&self, graph: ArenaGraph) -> [u8; 32] {
        let hash = hash_graph(&graph);

        if XTREME_ACTIVE.load(Ordering::Relaxed) {
            // Xtreme: bypass the multi-stage pool entirely.
            // Route directly to the dedicated single-core Xtreme worker via its
            // own injector.  The Xtreme worker runs the full pipeline inline and
            // deduplicates against the JIT cache itself.
            XTREME_INJECTOR.push(Task::Assemble { graph: Arc::new(graph), hash });
            return hash;
        }

        if JIT_CACHE.contains_key(&hash) {
            return hash;
        }
        self.injector.push(Task::Tile { graph: Arc::new(graph), hash });
        hash
    }
}

/// Adaptive idle strategy for regular pool worker threads.
///
/// Phases entered in order as the work-stealing queue stays empty:
///
/// | Phase  | Condition                       | Mechanism           | Rationale                               |
/// |--------|---------------------------------|---------------------|-----------------------------------------|
/// | Xtreme | `is_xtreme` (always)            | 1 ms sleep          | Release core; Xtreme worker handles it |
/// | 1      | `idle_count < SPIN_LIMIT`       | `spin_loop()` hints | ~40-cycle pause; wake within ns        |
/// | 2      | `< YIELD_LIMIT` or `is_hyper`   | `yield_now()`       | Cooperative; still responsive          |
/// | 3      | `≥ YIELD_LIMIT`, normal only    | Exponential sleep   | 1 → 2 → 4 → 8 → 16 ms; saves power    |
///
/// **Xtreme mode**: regular pool threads are parked — no Xtreme tasks reach
///   them (all work routes through `XTREME_INJECTOR`).  A 1 ms sleep keeps
///   them out of the way of the dedicated Xtreme core.
/// **Hyper mode**: thread never sleeps — oscillates between phases 1 and 2.
/// **Normal/Eager**: full three-phase graduated back-off.
///
/// `idle_count` resets to `0` whenever a task is dequeued.
#[cold]
#[inline(never)]
fn idle_backoff(idle_count: u64, is_hyper: bool, is_xtreme: bool) {
    /// `spin_loop()` iterations before backing off to yields.
    const SPIN_LIMIT: u64 = 200;
    /// Additional `yield_now()` calls after the spin phase ends.
    const YIELD_COUNT: u64 = 200;
    /// Total idle iterations before entering the exponential sleep phase.
    const YIELD_LIMIT: u64 = SPIN_LIMIT + YIELD_COUNT;
    /// Maximum exponent for sleep duration: 2^4 = 16 ms ceiling.
    const MAX_SLEEP_SHIFT: u64 = 4;

    if is_xtreme {
        // Xtreme — regular pool threads park themselves.
        // All Xtreme work goes to the dedicated single-core worker via
        // XTREME_INJECTOR; these threads have nothing to do.  Sleep briefly
        // so they stop competing with the Xtreme worker for scheduler slots.
        thread::sleep(std::time::Duration::from_millis(1));
    } else if idle_count < SPIN_LIMIT {
        // Phase 1 — hot spin: minimal latency for bursty incoming work.
        std::hint::spin_loop();
    } else if is_hyper || idle_count < YIELD_LIMIT {
        // Phase 2 — cooperative yield: share the CPU while staying responsive.
        // Hyper-mode threads never advance past this phase.
        thread::yield_now();
    } else {
        // Phase 3 — exponential back-off: conserve power when genuinely idle.
        // Each successive empty poll doubles the sleep up to MAX_SLEEP_SHIFT (16 ms).
        let shift = (idle_count - YIELD_LIMIT).min(MAX_SLEEP_SHIFT);
        thread::sleep(std::time::Duration::from_millis(1u64 << shift));
    }
}

fn worker_loop(
    local: Worker<Task>,
    global_stealers: Vec<Stealer<Task>>,
    injector: Arc<Injector<Task>>,
    active: Arc<AtomicBool>,
    role: &str,
) {
    let mut idle_count: u64 = 0;
    let mut current_is_hyper = false;
    
    #[cfg(windows)]
    use windows::Win32::System::Threading::{SetThreadPriority, GetCurrentThread, THREAD_PRIORITY_TIME_CRITICAL, THREAD_PRIORITY_NORMAL};

    while active.load(Ordering::Relaxed) {
        let mode = crate::get_execution_mode();
        // Xtreme is a strict superset of Hyper: both get TIME_CRITICAL threads.
        let is_hyper = mode == crate::ExecutionMode::Hyper
                    || mode == crate::ExecutionMode::Xtreme;
        // Fast-path bool — one Relaxed load vs. load+match on the full u8.
        let is_xtreme = XTREME_ACTIVE.load(Ordering::Relaxed);

        if is_hyper != current_is_hyper {
            current_is_hyper = is_hyper;
            #[cfg(windows)]
            unsafe {
                if is_hyper {
                    let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
                } else {
                    let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
                }
            }
        }

        let mut task = local.pop();

        if task.is_none() {
            loop {
                match injector.steal_batch_and_pop(&local) {
                    crossbeam_deque::Steal::Success(t) => {
                        task = Some(t);
                        break;
                    }
                    crossbeam_deque::Steal::Empty => break,
                    crossbeam_deque::Steal::Retry => continue,
                }
            }
        }

        // Assemblers steal from peer queues to keep all work-stealing queues drained.
        // (In Xtreme mode no tasks reach the regular pool at all, so this is a no-op.)
        if task.is_none() && role == "assembler" {
            for stealer in &global_stealers {
                loop {
                    match stealer.steal_batch_and_pop(&local) {
                        crossbeam_deque::Steal::Success(t) => {
                            task = Some(t);
                            break;
                        }
                        crossbeam_deque::Steal::Empty => break,
                        crossbeam_deque::Steal::Retry => continue,
                    }
                }
                if task.is_some() {
                    break;
                }
            }
        }

        if let Some(t) = task {
            idle_count = 0;
            process_task(t, &local, is_xtreme);
        } else {
            idle_count += 1;
            idle_backoff(idle_count, current_is_hyper, is_xtreme);
        }
    }
}

/// Called by `set_execution_mode` whenever the mode changes.
/// Ensures the Xtreme worker thread exists when Xtreme mode is first activated.
pub fn update_pool_mode() {
    if XTREME_ACTIVE.load(Ordering::Relaxed) {
        ensure_xtreme_worker_started();
    }
}

/// Spawns the Xtreme worker thread exactly once.
///
/// Uses `compare_exchange` so that concurrent callers are safe — only one
/// thread wins the race and spawns the OS thread.
fn ensure_xtreme_worker_started() {
    if XTREME_WORKER_STARTED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
        .is_ok()
    {
        // Select the last available logical core as the dedicated Xtreme core.
        // On typical systems core 0 handles many OS interrupt affinity assignments;
        // the last core is usually quieter.
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        let xtreme_core = core_ids.last().copied();

        thread::Builder::new()
            .name("numtoy-xtreme".into())
            .spawn(move || xtreme_worker_loop(xtreme_core))
            .expect("Failed to spawn Xtreme worker thread");
    }
}

/// The dedicated single-core Xtreme worker loop.
///
/// Behaviour summary:
/// - **Pinned** to the last available logical core via `core_affinity`.
/// - **`THREAD_PRIORITY_TIME_CRITICAL`** (Windows) while Xtreme mode is active,
///   `THREAD_PRIORITY_NORMAL` otherwise — so it doesn't hog the core when idle.
/// - **Hot-spins** on `XTREME_INJECTOR` while active; sleeps when mode is off.
/// - Calls `xtreme_run_inline` for every task — the full `Tile → Assemble`
///   pipeline in one synchronous call, no sub-queuing, no task re-push.
fn xtreme_worker_loop(core_id: Option<core_affinity::CoreId>) {
    // Pin this thread to the dedicated Xtreme core.
    if let Some(id) = core_id {
        core_affinity::set_for_current(id);

        // Linux only: push the isolated core into "performance" scaling governor
        // so the CPU frequency ramp-up happens immediately rather than after the
        // OS governor notices sustained load.  A best-effort write — non-root
        // processes usually lack permission to write this file, so we silently
        // ignore any error rather than panicking.
        #[cfg(target_os = "linux")]
        {
            let path = format!(
                "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor",
                id.id
            );
            let _ = std::fs::write(&path, "performance\n");
        }
    }

    #[cfg(windows)]
    use windows::Win32::System::Threading::{
        SetThreadPriority, GetCurrentThread,
        THREAD_PRIORITY_TIME_CRITICAL, THREAD_PRIORITY_NORMAL,
    };

    let mut priority_elevated = false;

    loop {
        let is_xtreme = XTREME_ACTIVE.load(Ordering::Relaxed);

        // Dynamically adjust thread priority as mode changes.
        if is_xtreme != priority_elevated {
            priority_elevated = is_xtreme;
            #[cfg(windows)]
            unsafe {
                if is_xtreme {
                    let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
                } else {
                    let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
                }
            }
        }

        if !is_xtreme {
            // Mode is off — release the core entirely.
            thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }

        // Hot-drain the Xtreme injector.
        match XTREME_INJECTOR.steal() {
            crossbeam_deque::Steal::Success(task) => {
                xtreme_run_inline(task);
            }
            crossbeam_deque::Steal::Empty => {
                // Nothing in the queue — spin with a pause hint.
                std::hint::spin_loop();
            }
            crossbeam_deque::Steal::Retry => {
                // Transient contention — retry immediately.
            }
        }
    }
}

/// Execute the full compilation pipeline for one task, inline and synchronously.
///
/// Replaces the three-stage `Tile → Fuse → Assemble` queue round-trips with a
/// single function call.  Called only from `xtreme_worker_loop`; never re-pushes
/// sub-tasks to any queue.
fn xtreme_run_inline(task: Task) {
    let (graph, hash) = match task {
        Task::Tile     { graph, hash }
        | Task::Fuse   { graph, hash }
        | Task::Assemble { graph, hash } => (graph, hash),
    };

    // Final dedup: another thread (or a previous Xtreme submit) may have already
    // compiled and cached this graph.
    if JIT_CACHE.contains_key(&hash) {
        return;
    }

    // Prefetch graph nodes into L1 before the JIT compiler begins reading them.
    // Hides any DRAM latency for graphs that have been evicted from cache.
    #[cfg(target_arch = "x86_64")]
    if !graph.nodes.is_empty() {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        let base = graph.nodes.as_ptr() as *const i8;
        let total_bytes = graph.nodes.len() * std::mem::size_of::<crate::graph::Node>();
        // Cover up to 512 bytes (8 × 64-byte cache lines).
        let prefetch_end = total_bytes.min(512);
        let mut off = 0usize;
        while off < prefetch_end {
            unsafe { _mm_prefetch(base.add(off), _MM_HINT_T0); }
            off += 64;
        }
    }

    if let Some(exec_fn) = compile_packed_kernel(&graph) {
        JIT_CACHE.insert(hash, exec_fn);
    }
}

fn process_task(task: Task, local: &Worker<Task>, is_xtreme: bool) {
    match task {
        Task::Tile { graph, hash } => {
            if is_xtreme {
                // Xtreme: skip the Fuse optimization pass entirely.
                // Go straight to machine-code assembly — no stabilization round-trips.
                local.push(Task::Assemble { graph, hash });
            } else {
                local.push(Task::Fuse { graph, hash });
            }
        }
        Task::Fuse { graph, hash } => {
            if graph.is_stabilized {
                local.push(Task::Assemble { graph, hash });
            } else {
                // Optimization pass: mark as stabilized to avoid infinite loops.
                let mut new_graph = (*graph).clone();
                new_graph.is_stabilized = true;
                local.push(Task::Fuse { graph: Arc::new(new_graph), hash });
            }
        }
        Task::Assemble { graph, hash } => {
            if JIT_CACHE.contains_key(&hash) {
                return;
            }

            // Xtreme: prefetch the first N cache lines of the graph's node array
            // into L1 before the JIT compiler begins reading them.  Hides DRAM
            // latency behind any other in-flight work on the core.
            #[cfg(target_arch = "x86_64")]
            if is_xtreme && !graph.nodes.is_empty() {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                let base = graph.nodes.as_ptr() as *const i8;
                let total_bytes = graph.nodes.len() * std::mem::size_of::<crate::graph::Node>();
                // Prefetch up to 512 bytes (8 × 64-byte cache lines).
                let prefetch_end = total_bytes.min(512);
                let mut off = 0usize;
                while off < prefetch_end {
                    unsafe { _mm_prefetch(base.add(off), _MM_HINT_T0); }
                    off += 64;
                }
            }

            if let Some(exec_fn) = compile_packed_kernel(&graph) {
                JIT_CACHE.insert(hash, exec_fn);
            }
        }
    }
}
