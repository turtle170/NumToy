use std::sync::{Arc, Mutex, Condvar};
use std::thread;
use crossbeam_deque::{Worker, Stealer, Injector};
use crate::graph::ArenaGraph;
use crate::cache::{JIT_CACHE, hash_graph, ExecFn};
use crate::fuser::compile_packed_kernel;
use std::sync::atomic::{AtomicBool, Ordering};

pub enum Task {
    Tile { graph: Arc<ArenaGraph>, hash: [u8; 32] },
    Fuse { graph: Arc<ArenaGraph>, hash: [u8; 32] },
    Assemble { graph: Arc<ArenaGraph>, hash: [u8; 32] },
}

use once_cell::sync::Lazy;

pub static GLOBAL_POOL: Lazy<ThreadPool> = Lazy::new(|| ThreadPool::new());

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

            thread::spawn(move || {
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
        if JIT_CACHE.contains_key(&hash) {
            return hash;
        }
        self.injector.push(Task::Tile { graph: Arc::new(graph), hash });
        hash
    }
}

fn worker_loop(
    local: Worker<Task>,
    global_stealers: Vec<Stealer<Task>>,
    injector: Arc<Injector<Task>>,
    active: Arc<AtomicBool>,
    role: &str,
) {
    let mut idle_count = 0;
    while active.load(Ordering::Relaxed) {
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
            process_task(t, &local);
        } else {
            idle_count += 1;
            if idle_count > 100 {
                thread::yield_now();
            }
        }
    }
}

fn process_task(task: Task, local: &Worker<Task>) {
    match task {
        Task::Tile { graph, hash } => {
            // Tiler determines tile size, but for now we just push to Fuse
            local.push(Task::Fuse { graph, hash });
        }
        Task::Fuse { graph, hash } => {
            // If already stabilized by a previous pass, assemble
            if graph.is_stabilized {
                local.push(Task::Assemble { graph, hash });
            } else {
                // Mock optimization pass: mark as stabilized to avoid infinite loops
                let mut new_graph = (*graph).clone();
                new_graph.is_stabilized = true;
                local.push(Task::Fuse { graph: Arc::new(new_graph), hash });
            }
        }
        Task::Assemble { graph, hash } => {
            if JIT_CACHE.contains_key(&hash) {
                return;
            }
            // Assemble into machine code
            if let Some(exec_fn) = compile_packed_kernel(&graph) {
                JIT_CACHE.insert(hash, exec_fn);
            }
        }
    }
}
