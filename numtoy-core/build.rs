use std::process::Command;
use std::path::Path;
use std::env;

fn main() {
    // Re-run this build script if build.rs or the Zig code changes
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../numtoy-hardware/src/main.zig");
    println!("cargo:rerun-if-changed=../numtoy-hardware/build.zig");

    let cargo_manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let zig_dir = Path::new(&cargo_manifest_dir).parent().unwrap().join("numtoy-hardware");

    // 1. Find Zig executable
    let zig_bin = env::var("ZIG").unwrap_or_else(|_| "zig".to_string());
    let test_zig = Command::new(&zig_bin)
        .arg("version")
        .status();
    if test_zig.is_err() || !test_zig.unwrap().success() {
        panic!(
            "Could not find or execute `zig`. Please ensure Zig is installed and on your PATH, or set the ZIG environment variable to point to the Zig executable."
        );
    }

    // Determine target triple from CARGO build environment
    let target = env::var("TARGET").unwrap_or_else(|_| "x86_64-pc-windows-msvc".to_string());
    let mut args = vec!["build".to_string(), "-Doptimize=ReleaseFast".to_string()];

    let zig_target = if target.contains("aarch64-unknown-linux-gnu") {
        Some("aarch64-linux-gnu")
    } else if target.contains("x86_64-unknown-linux-gnu") {
        Some("x86_64-linux-gnu")
    } else if target.contains("aarch64-apple-darwin") {
        Some("aarch64-macos")
    } else if target.contains("x86_64-apple-darwin") {
        Some("x86_64-macos")
    } else if target.contains("x86_64-pc-windows-msvc") {
        Some("x86_64-windows-msvc")
    } else if target.contains("x86_64-pc-windows-gnu") {
        Some("x86_64-windows-gnu")
    } else {
        None
    };

    if let Some(zt) = zig_target {
        args.push(format!("-Dtarget={}", zt));
    }

    // Run zig build in numtoy-hardware
    let status = Command::new(&zig_bin)
        .args(&args)
        .current_dir(&zig_dir)
        .status()
        .expect("Failed to execute zig build");
    assert!(status.success(), "Zig compilation failed");

    // 2. Link the generated static library
    let lib_dir = zig_dir.join("zig-out").join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.to_str().unwrap());
    println!("cargo:rustc-link-lib=static=numtoy_hardware");

    // Platform-specific linker flags for downstream targets
    if target.contains("apple") || target.contains("darwin") {
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
    }

    // 3. Write C header for the C++ wrapper
    // We maintain numtoy.h manually here rather than via cbindgen, because
    // the crate's internal modules contain complex Rust generics that cbindgen
    // cannot parse. The public extern "C" API is small and stable.
    let cpp_dir = Path::new(&cargo_manifest_dir).parent().unwrap().join("numtoy-cpp");
    std::fs::create_dir_all(&cpp_dir).ok();

    let header = r#"/* WARNING: This file was hand-generated from numtoy-core/build.rs. */
#ifndef NUMTOY_H
#define NUMTOY_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HardwareEngine HardwareEngine;
typedef struct Expr Expr;
typedef struct CppTensor CppTensor;

/* ── Engine ─────────────────────────────────────────── */
HardwareEngine *nt_engine_new(void);
void            nt_engine_free(HardwareEngine *engine);

/* ── Scalar Expressions ─────────────────────────────── */
Expr *nt_expr_new_var(HardwareEngine *engine,
                      uintptr_t id,
                      const char *name,
                      uint32_t dtype_val,
                      uint32_t bits,
                      const double *values,
                      uintptr_t count,
                      uint32_t scale);

Expr *nt_expr_new_const(double val);

Expr *nt_expr_add(Expr *left, Expr *right);
Expr *nt_expr_sub(Expr *left, Expr *right);
Expr *nt_expr_mul(Expr *left, Expr *right);
Expr *nt_expr_div(Expr *left, Expr *right);

/* Returns symbolic gradient d(expr)/d(wrt_id) */
Expr *nt_expr_grad(Expr *expr, uintptr_t wrt_id);

void nt_expr_free(Expr *expr);

/* device_str: NULL/"cpu" = Cranelift JIT, "gpu" = WebGPU */
Expr *nt_expr_execute(HardwareEngine *engine, Expr *expr, const char *device_str);

uintptr_t nt_expr_unpack(HardwareEngine *engine,
                          Expr *expr,
                          double *out_values,
                          uintptr_t max_count);

/* ── Tensors ─────────────────────────────────────────── */
CppTensor *nt_tensor_new(HardwareEngine *engine,
                         uintptr_t id,
                         const char *name,
                         uint32_t dtype_val,
                         uint32_t bits,
                         const double *values,
                         uintptr_t count,
                         uint32_t scale,
                         const uintptr_t *shape_ptr,
                         uintptr_t shape_len);

void       nt_tensor_free(CppTensor *t);
CppTensor *nt_tensor_grad(CppTensor *t, uintptr_t wrt_id);
CppTensor *nt_tensor_execute(HardwareEngine *engine,
                              CppTensor *t,
                              const char *device_str);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NUMTOY_H */
"#;

    std::fs::write(cpp_dir.join("numtoy.h"), header)
        .expect("Unable to write numtoy.h");
}
