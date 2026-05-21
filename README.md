# NumToy [![CI](https://github.com/turtle170/NumToy/actions/workflows/ci.yml/badge.svg)](https://github.com/turtle170/NumToy/actions/workflows/ci.yml)

A compiler-grade, multi-language numerical engine with hybrid Rust + Zig architecture.

## Architecture

| Component | Language | Role |
|---|---|---|
| `numtoy-hardware/` | Zig | Low-level memory engine – arbitrary-bit packing, SIMD tiling, ArenaAllocator |
| `numtoy-core/` | Rust | IR, two-stage fuser (Cranelift JIT + WebGPU), auto-diff, Python/C ABI |
| `numtoy-python/` | Python + PyO3 | Maturin wheel – `@nt.compile` decorator, `Tensor` API |
| `numtoy-cpp/` | C++ | Header-only RAII wrapper – operator overloads, lazy graph building |

## Platform Support

| Operating System | Architecture | Vector Engine | GPU Backend |
|---|---|---|---|
| **Windows** | `x86_64` | SSE2 / AVX | WebGPU (DX12 / Vulkan) |
| **Linux** | `x86_64`, `ARM64` (`aarch64`) | SSE2 / AVX, ARM NEON | WebGPU (Vulkan) |
| **macOS** | `x86_64`, Apple Silicon (`aarch64`) | SSE2, ARM NEON | WebGPU (Metal) |


## Features

- **Custom IR** – lazy evaluation graph (`Variable`, `Constant`, `Add`, `Mul`)
- **Two-stage fuser** – Stage 1 micro-kernel fusion; Stage 2 Cranelift JIT (CPU) or WebGPU (GPU)
- **AdaptableFloat Sign Stealer** – automatically reallocates the sign bit to the mantissa for all-positive datasets
- **Custom bit-width types** – `Float(bits)`, `Int(bits)`, `DynamicFloat`, `FloatingInt` (fixed-point)
- **Broadcast tiling** – SIMD vectorization for custom-width arrays via Zig's `@Vector`
- **Auto-diff** – reverse-mode automatic differentiation through the IR graph
- **Unified memory tensors** – multi-dimensional `Tensor` type with shape/stride/broadcast

## Quick Start

### Python
```python
import numtoy as nt

engine = nt.Engine()

@nt.compile
def my_kernel(x, y):
    return (x + y) * 2.5

result = my_kernel(engine, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
print(result.unpack(engine))  # [12.5, 17.5, 22.5]

# GPU execution
result_gpu = my_kernel(engine, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], device="gpu")
```

### C++
```cpp
#include "numtoy.hpp"

numtoy::Engine engine;
auto x = numtoy::Expr::var(engine, 1, "x", numtoy::DataType::Float, 32, {1.0, 2.0, 3.0});
auto y = numtoy::Expr::var(engine, 2, "y", numtoy::DataType::Float, 32, {4.0, 5.0, 6.0});
auto z = (x + y) * numtoy::Expr::constant(2.5);
auto result = z.execute(engine, "cpu");
```

## Building

### Requirements
- Rust (nightly)
- Zig 0.14+
- Python 3.10+ with `maturin`

### Python Wheel
```bash
cd numtoy-python
maturin build
pip install ../numtoy-core/target/wheels/numtoy-*.whl
```

### C++ Test
```bash
cd numtoy-cpp
zig c++ test_numtoy.cpp -I. -L../numtoy-core/target/debug -lnumtoy_core -std=c++17 -o test_numtoy.exe
cp ../numtoy-core/target/debug/numtoy_core.dll .
./test_numtoy.exe
```
