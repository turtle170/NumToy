# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Is NumToy

NumToy is a compiler-grade numerical engine with a **Rust core + Zig hardware layer + Python/C++ frontends**. It builds lazy computation graphs, JIT-compiles them to native code via Cranelift, and supports custom bit-width types (e.g., 5-bit floats, 3-bit ints), automatic differentiation, GPU execution via WebGPU, and multi-dimensional tensor operations.

---

## Build Commands

### Rust core (from `numtoy-core/`)
```powershell
cargo build --target x86_64-pc-windows-msvc
cargo build --release --target x86_64-pc-windows-msvc
```

### Python wheel (from `numtoy-python/`)
```powershell
maturin build --release --out dist/
maturin develop   # install into current venv in-place (dev workflow)
```

### Zig hardware engine (from `numtoy-hardware/`)
```powershell
zig build
zig build -Doptimize=ReleaseFast
```

### C++ tests
```powershell
zig c++ test_numtoy.cpp -I. -L..\numtoy-core\target\debug -lnumtoy_core -std=c++17 -o test_numtoy.exe
```

---

## Test Commands

### Rust unit + integration tests (from `numtoy-core/`)
```powershell
cargo test --target x86_64-pc-windows-msvc
cargo test <test_name> --target x86_64-pc-windows-msvc   # run a single test
cargo test -- --nocapture                                  # show println! output
```
Key integration tests live at the bottom of `numtoy-core/src/lib.rs`:
- `test_end_to_end_jit_compilation`
- `test_auto_diff_product_rule`
- `test_tensor_broadcast_add`

### Zig tests (from `numtoy-hardware/`)
```powershell
zig test src/main.zig
```

### Python tests (repo root)
```powershell
python test_numtoy.py       # full feature test suite
python test_scalable.py     # scalable / dynamic bit-width types
python numtoy-python/test_gpu.py
python numtoy-python/test_matmul.py
python numtoy-python/test_modes.py
```

### Lint
```powershell
cargo clippy --target x86_64-pc-windows-msvc -- -D warnings
zig fmt src/                # Zig formatter
```

---

## Architecture Overview

```
numtoy-python/     Python API (PyO3 bindings via maturin)
numtoy-core/       Rust engine (IR, JIT, GPU, auto-diff, C ABI)
numtoy-hardware/   Zig engine (bit-packing, SIMD tiling, arena allocator)
numtoy-cpp/        C++ header-only RAII wrapper
```

### Data flow

1. **Build a lazy graph** — Python/C++ calls produce `Expr` nodes (`Variable`, `Constant`, `Add`, `Mul`, `Sub`, `Div`, `DequantizeMatmul`) that form a tree. No computation runs yet.
2. **Graph flattening** — `ArenaGraph` (graph.rs) flattens the recursive `Expr` tree into an indexed `Vec<Node>` for efficient traversal.
3. **Two-stage fuser** (fuser.rs):
   - *Stage 1*: Traverses the flat graph and generates bit-packing micro-kernel helper code (`helper_load_u64`, `helper_store_u64`, `helper_decode_to_f64`).
   - *Stage 2*: Cranelift JIT compiles those helpers to native x86_64 / ARM64 machine code.
4. **JIT cache** — `JIT_CACHE` (cache.rs) is a `DashMap` keyed on Blake3 hashes of graph structure + data types + shapes. Identical subgraphs skip recompilation.
5. **Thread pool** — `ThreadPool` (pool.rs) uses crossbeam work-stealing queues with three thread roles: *Tilers → Fusers → Assemblers*. In *Hyper* mode, Windows thread priorities are elevated to `THREAD_PRIORITY_TIME_CRITICAL`.
6. **GPU path** — `execute_gpu()` (gpu.rs) generates a WGSL compute shader from the expression tree, uploads bit-packed buffers, runs on wgpu, and repacks the f32 result.
7. **Hardware engine** — FFI calls into the Zig library (`nt_pack`, `nt_unpack`, `nt_tile`) for all bit-level memory operations.

### Key source files

| File | Role |
|---|---|
| `numtoy-core/src/lib.rs` | PyO3 bindings (`PyEngine`, `PyExpr`, `PyTensor`), C-ABI exports (`nt_engine_*`), global execution-mode flag |
| `numtoy-core/src/types.rs` | `DataType` enum (Float, Int, DynamicFloat, FloatingInt, ScalableInt, ScalableFloat) and `Scalar` encoder/decoder for every custom width |
| `numtoy-core/src/ir.rs` | `Expr` tree, `PackedBuffer` (Memory or Mmap), symbolic auto-diff (`.grad(wrt_id)`) |
| `numtoy-core/src/graph.rs` | `ArenaGraph` — flattens Expr tree into indexed nodes |
| `numtoy-core/src/fuser.rs` | Two-stage JIT: micro-kernel generation → Cranelift compilation; `helper_dequantize_matmul()` |
| `numtoy-core/src/cache.rs` | Blake3 graph hashing + `JIT_CACHE` DashMap |
| `numtoy-core/src/pool.rs` | Crossbeam work-stealing thread pool; Hyper mode priority support |
| `numtoy-core/src/array.rs` | `NumToyArray`: shape/stride metadata, broadcast, reshape, slice, transpose |
| `numtoy-core/src/gpu.rs` | WebGPU async pipeline, dynamic WGSL shader codegen |
| `numtoy-core/src/calculator.rs` | Background thread: memory-latency vs. arithmetic heuristics for Eager mode rewrites |
| `numtoy-core/src/hardware.rs` | FFI bridge to Zig (`HardwareEngine`) |
| `numtoy-python/numtoy/__init__.py` | High-level Python API: `Engine`, `Expr`, `Tensor`, `@compile`, `mode()` context manager |

---

## Type System

`DataType` variants (defined in `types.rs`):

| Variant | Description |
|---|---|
| `Float(bits)` | IEEE-like float; `bits=32` or `bits=64` use hardware types |
| `Int(bits)` | Signed integer; `bits=16/32` use hardware |
| `DynamicFloat` | Auto-selects precision at encode time |
| `FloatingInt` | Fixed-point: integer with a scale factor |
| `ScalableInt(bits)` | Dynamic mantissa allocation up to `bits` |
| `ScalableFloat(bits, exp_bits)` | Dynamic exponent + mantissa split within `bits` |

**AdaptableFloat Sign Stealer**: when all values in a buffer are non-negative, `should_steal_sign()` reallocates the sign bit to mantissa, gaining one extra precision bit automatically.

---

## Execution Modes

Set via `nt.mode("hyper")` context manager in Python, or `set_execution_mode()` in Rust:

| Mode | Behavior |
|---|---|
| `default` | Standard lazy eval + JIT |
| `eager` | Immediate execution; Calculator thread applies heuristic rewrites |
| `hyper` | `THREAD_PRIORITY_TIME_CRITICAL` on Windows; max throughput |

---

## Auto-Differentiation

Symbolic reverse-mode AD lives entirely in `ir.rs` — `Expr::grad(wrt_id)` recursively builds a new `Expr` tree representing the derivative without executing anything. Rules implemented: identity (∂var/∂var = 1), constant (0), sum/difference, product rule, quotient rule. The gradient `Expr` is then JIT-compiled the same way as any other expression.

---

## Python API Quick Reference

```python
import numtoy as nt

# Basic expression
engine = nt.Engine()
x = nt.Expr.var(engine, "x", [1.0, 2.0], dtype="float32")
y = nt.Expr.var(engine, "y", [3.0, 4.0], dtype="float32")
result = ((x + y) * 2.0).execute(engine).unpack(engine)

# Auto-diff
dx = (x * y).grad("x")           # returns an Expr tree

# Decorator
@nt.compile
def kernel(x, y): return (x + y) * 2.0
result = kernel(engine, [1.0, 2.0], [3.0, 4.0])

# Execution mode
with nt.mode("hyper"):
    result = expr.execute(engine)

# Tensors
t = nt.Tensor.from_values(engine, [1.0,2.0,3.0,4.0,5.0,6.0], [3,2])
t2 = t.broadcast_to(engine, [3,2]).execute(engine)
```

---

## CI / Release

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs a **5-platform matrix**: Windows x86_64, Linux x86_64, Linux ARM64, macOS x86_64, macOS ARM64. Steps include Zig 0.14.1 install, Rust toolchain setup, maturin wheel build, cargo test, and NEON instruction-emission checks on ARM64. Wheels are automatically published to PyPI on `v*` tags.
