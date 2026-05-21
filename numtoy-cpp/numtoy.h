/* WARNING: This file was hand-generated from numtoy-core/build.rs. */
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
