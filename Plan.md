Hey Antigravity! Let's scaffold and build "NumToy"—a compiler-grade, multi-language numerical engine. It needs a hybrid architecture featuring a Rust compiler core (for graph management, lazy evaluation, and fuser orchestration) combined with Zig for low-level, high-efficiency hardware primitives and custom-bit representations.

Please implement the workspace with the following plan:

1. **Repository Layout**:
   - `numtoy-core/` (Rust crate, configured as a cdylib/rlib)
   - `numtoy-hardware/` (Zig library)
   - `numtoy-python/` (Python project wrapper using Maturin)
   - `numtoy-cpp/` (C++ modern header-only wrapper)

2. **Core Requirements to Implement**:
   - **Custom Intermediate Representation (IR)**: In `numtoy-core`, build a Rust enum-based IR structure for lazy evaluation graph building (`Variable`, `Constant`, `Add`, `Mul`). Support custom byte and bit-width types: dynamic `Float(bits)`, `Int(bits)`, an adaptable dynamic-bit float, and a `FloatingInt` (Fixed-point representation where numbers like 12.1 are stored as 121 with an explicit `scale` tracking factor).
   - **Two-Stage Fuser**: 
     - *Stage 1 (Micro-kernel generation)*: Write an engine optimizer pass that traverses the lazy IR graph, eliminates redundant allocations/reads, and collapses chained element-wise array operations into single fused loop blocks.
     - *Stage 2 (JIT Compilation)*: Use the `cranelift` crate ecosystem (`cranelift`, `cranelift-module`, `cranelift-jit`) to compile these fused micro-kernels directly into raw, hardware-optimized CPU machine code at runtime.
   - **Bit Packing & Memory Layout**: In `numtoy-hardware`, use Zig to manage memory directly. Take advantage of Zig's arbitrary-width integer primitives (like `u3` or `u11`) to store variables in raw, densely-packed byte streams. Implement a stateful tracking engine handle that leverages a Zig `ArenaAllocator` so thousands of small arrays and custom types can be freed simultaneously without memory leaks.
   - **Multi-Language Interfaces**:
     - Integrate `PyO3` and a `pyproject.toml` using `maturin` so that NumToy can be compiled straight into a highly distributable Python Wheel. Expose a Python `@nt.compile` decorator that converts native python functions into your lazy IR graph.
     - Add `cbindgen` to automatically compile a clean C-compatible `numtoy.h` header interface.
     - Create modern C++ wrapper classes in `numtoy-cpp` that overload operators (`+`, `*`, `[]`) to handle lazy expression graph generation behind the scenes, presenting a clean experience for native developers.

Go ahead and generate the configuration files, cross-compilation pipeline steps, and the foundational implementation for each directory.