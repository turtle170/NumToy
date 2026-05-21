/* WARNING: This file was hand-generated from numtoy-core/build.rs. */
#ifndef NUMTOY_H
#define NUMTOY_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HardwareEngine HardwareEngine;
typedef struct EngineOpaque {} EngineOpaque;
typedef struct Expr Expr;

HardwareEngine *nt_engine_new(void);
void nt_engine_free(HardwareEngine *engine);

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
Expr *nt_expr_mul(Expr *left, Expr *right);

void nt_expr_free(Expr *expr);

/* device_str: NULL or "cpu" for Cranelift JIT, "gpu" for WebGPU */
Expr *nt_expr_execute(HardwareEngine *engine, Expr *expr, const char *device_str);

uintptr_t nt_expr_unpack(HardwareEngine *engine,
                         Expr *expr,
                         double *out_values,
                         uintptr_t max_count);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NUMTOY_H */
