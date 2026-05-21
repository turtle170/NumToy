"""
NumToy Python Integration Tests
Covers: float32, int16, floating-point, @compile decorator,
        AdaptableFloat Sign Stealer, GPU execution,
        auto-diff (product rule), Tensor broadcast, Tensor reshape.
"""
import numtoy as nt

def test_numtoy_python():
    engine = nt.Engine()

    # ── Float32 round-trip ────────────────────────────────────────────────────
    print("Initializing NumToy Engine...")
    print("Creating variables...")
    x = nt.Expr.var(engine, "x", [1.0, 2.0, 3.0])
    y = nt.Expr.var(engine, "y", [4.0, 5.0, 6.0])
    print("Building expression tree: z = (x + y) * 2.5")
    z = (x + y) * 2.5
    print("Compiling & Executing expression tree...")
    res = z.execute(engine)
    print("Unpacking results...")
    vals = res.unpack(engine)
    print(f"Unracked Results: {vals}")
    expected = [12.5, 17.5, 22.5]
    assert vals == expected, f"Float32 mismatch: {vals}"
    print("Float32 test passed successfully!\n")

    # ── Int16 ─────────────────────────────────────────────────────────────────
    print("Testing packed Int16...")
    a = nt.Expr.var(engine, "a", [5.0, -10.0, 15.0], dtype="int", bits=16)
    b = nt.Expr.var(engine, "b", [10.0, 20.0, 30.0], dtype="int", bits=16)
    ab = (a * b).execute(engine)
    ab_vals = ab.unpack(engine)
    print(f"Int16 Multiplication Results: {ab_vals}")
    assert ab_vals == [50.0, -200.0, 450.0], f"Int16 mismatch: {ab_vals}"
    print("Int16 test passed successfully!\n")

    # ── FloatingInt ───────────────────────────────────────────────────────────
    print("Testing FloatingInt (fixed point mapped to integer) with scale 2 (100x)...")
    p = nt.Expr.var(engine, "p", [1.0, 2.0, 3.0], dtype="floating_int", scale=2)
    q = nt.Expr.var(engine, "q", [1.0, 2.0, 3.0], dtype="floating_int", scale=2)
    pq = (p + q).execute(engine)
    pq_vals = pq.unpack(engine)
    print(f"FloatingInt Addition Results: {pq_vals}")
    assert pq_vals == [2.0, 4.0, 6.0], f"FloatingInt mismatch: {pq_vals}"
    print("FloatingInt test passed successfully!\n")

    # ── @compile decorator ────────────────────────────────────────────────────
    print("Testing @numtoy.compile decorator...")
    @nt.compile
    def my_kernel(x, y):
        return (x + y) * 3.0

    res_expr = my_kernel(engine, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    res_vals = res_expr.unpack(engine)
    print(f"Decorator Result: {res_vals}")
    assert res_vals == [15.0, 21.0, 27.0], f"Decorator mismatch: {res_vals}"
    print("Decorator test passed successfully!\n")

    # ── Sign Stealer ──────────────────────────────────────────────────────────
    print("Testing AdaptableFloat Sign Stealer (all non-negative -> extra mantissa bit)...")
    xp = nt.Expr.var(engine, "xpos", [1.5, 3.0, 4.5])
    yp = nt.Expr.var(engine, "ypos", [1.5, 3.0, 4.5])
    res_pos = (xp + yp).execute(engine, device="cpu")
    vals_pos = res_pos.unpack(engine)
    print(f"Sign Stealer Results (all positive): {vals_pos}")
    for got, exp in zip(vals_pos, [3.0, 6.0, 9.0]):
        assert abs(got - exp) < 0.2, f"Sign Stealer mismatch: {got} vs {exp}"
    print("Sign Stealer test passed!\n")

    # ── GPU execution ─────────────────────────────────────────────────────────
    print("Testing GPU execution via WebGPU (with CPU fallback if no GPU)...")
    @nt.compile
    def gpu_kernel(x, y):
        return (x + y) * 2.0

    gpu_result = gpu_kernel(engine, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], device="gpu")
    gpu_vals = gpu_result.unpack(engine)
    print(f"GPU Kernel Result: {gpu_vals}")
    for got, exp in zip(gpu_vals, [10.0, 14.0, 18.0]):
        assert abs(got - exp) < 0.5, f"GPU mismatch: {got} vs {exp}"
    print("GPU execution test passed!\n")

    # ── Auto-diff (product rule) ──────────────────────────────────────────────
    print("Testing Auto-Diff: d/dx [(x+y)*x] at x=2, y=3 => 2*x+y = 7 ...")
    xd = nt.Expr.var(engine, "xd", [2.0])
    yd = nt.Expr.var(engine, "yd", [3.0])
    zd = (xd + yd) * xd
    grad_expr = zd.grad(xd).execute(engine)
    grad_val = grad_expr.unpack(engine)[0]
    print(f"d(z)/d(x) = {grad_val}  (expected ~7.0)")
    assert abs(grad_val - 7.0) < 0.5, f"Auto-diff mismatch: {grad_val}"
    print("Auto-diff test passed!\n")

    # ── Tensor broadcast add ──────────────────────────────────────────────────
    print("Testing Tensor broadcast: [3,1] + [1,2] -> [3,2] ...")
    ta = nt.Tensor.from_values(engine, [1.0, 2.0, 3.0], [3, 1])
    tb = nt.Tensor.from_values(engine, [10.0, 20.0], [1, 2])
    tc_flat = (ta + tb).numpy(engine)
    print(f"Broadcast result (nested): {tc_flat}")
    expected_bc = [[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]]
    for row_got, row_exp in zip(tc_flat, expected_bc):
        for got, exp in zip(row_got, row_exp):
            assert abs(got - exp) < 0.5, f"Broadcast mismatch: {got} vs {exp}"
    print("Tensor broadcast test passed!\n")

    # ── Tensor reshape ────────────────────────────────────────────────────────
    print("Testing Tensor reshape: [2,3] -> [3,2] ...")
    tr = nt.Tensor.from_values(engine, list(range(6)), [2, 3])
    tr2 = tr.reshape([3, 2]).execute(engine)
    flat2 = tr2.numpy(engine)
    print(f"Reshape result: {flat2}")
    assert flat2 == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], f"Reshape mismatch: {flat2}"
    print("Tensor reshape test passed!\n")

    print("ALL PYTHON BINDINGS TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_numtoy_python()
