"""
NumToy Scalable and Arbitrary Precision (Infinite Width) Integration Tests
"""
import numtoy as nt
import math

def test_scalable_and_infinite_widths():
    engine = nt.Engine()

    print("=== Testing ScalableInt ===")
    # Magnitude of scalable_int scales based on the values.
    a = nt.Expr.var(engine, "a", [5.0, -10.0, 15.0], dtype="scalable_int")
    b = nt.Expr.var(engine, "b", [2.0, 3.0, 4.0], dtype="scalable_int")
    
    # 5*2=10, -10*3=-30, 15*4=60
    c = (a * b).execute(engine)
    c_vals = c.unpack(engine)
    print(f"ScalableInt Product: {c_vals}")
    assert c_vals == [10.0, -30.0, 60.0], f"ScalableInt mismatch: {c_vals}"

    print("=== Testing ScalableFloat ===")
    # Combines AdaptableFloat and ScalableInt's scalability
    x = nt.Expr.var(engine, "x", [1.5, 32.25, 0.125], dtype="scalable_float")
    y = nt.Expr.var(engine, "y", [2.0, 0.5, 8.0], dtype="scalable_float")
    
    xy = (x * y).execute(engine)
    xy_vals = xy.unpack(engine)
    print(f"ScalableFloat Product: {xy_vals}")
    expected_xy = [3.0, 16.125, 1.0]
    for got, exp in zip(xy_vals, expected_xy):
        assert abs(got - exp) < 1e-5, f"ScalableFloat mismatch: {got} vs {exp}"

    print("=== Testing Custom 128-bit Integers (Multi-Limb CPU Fallback) ===")
    # 2**40 fits precisely in f64
    val = float(2**40)
    # Using bits=64 (will overflow to 0 because lower 64 bits of 2**80 are 0)
    x64 = nt.Expr.var(engine, "x64", [val], dtype="int", bits=64)
    y64 = nt.Expr.var(engine, "y64", [val], dtype="int", bits=64)
    prod64 = (x64 * y64).execute(engine)
    res64 = prod64.unpack(engine)[0]
    print(f"64-bit Int Product of 2**40 * 2**40: {res64}")
    assert res64 == 9223372036854775807.0, f"Expected 64-bit overflow to saturate at i64::MAX, got {res64}"

    # Using bits=128 (will NOT overflow, limb interpreter splits into two 64-bit limbs)
    x128 = nt.Expr.var(engine, "x128", [val], dtype="int", bits=128)
    y128 = nt.Expr.var(engine, "y128", [val], dtype="int", bits=128)
    prod128 = (x128 * y128).execute(engine)
    res128 = prod128.unpack(engine)[0]
    print(f"128-bit Int Product of 2**40 * 2**40: {res128}")
    expected_128 = float(2**80)
    assert res128 == expected_128, f"Expected 2**80 ({expected_128}), got {res128}"

    print("=== Testing Custom 256-bit Integers ===")
    # 2**50 fits precisely in f64
    val50 = float(2**50)
    x256 = nt.Expr.var(engine, "x256", [val50], dtype="int", bits=256)
    y256 = nt.Expr.var(engine, "y256", [val50], dtype="int", bits=256)
    # 2**50 * 2**50 * 2**50 = 2**150
    prod256 = (x256 * y256 * x256).execute(engine)
    res256 = prod256.unpack(engine)[0]
    print(f"256-bit Int Product of 2**50 * 2**50 * 2**50: {res256}")
    expected_256 = float(2**150)
    assert res256 == expected_256, f"Expected 2**150 ({expected_256}), got {res256}"

    print("=== Testing Custom 128-bit Floats (Multi-Limb CPU Fallback) ===")
    xf128 = nt.Expr.var(engine, "xf128", [1.25, 2.5], dtype="float", bits=128)
    yf128 = nt.Expr.var(engine, "yf128", [4.0, 2.0], dtype="float", bits=128)
    prodf128 = (xf128 * yf128).execute(engine)
    resf128 = prodf128.unpack(engine)
    print(f"128-bit Float Product: {resf128}")
    assert resf128 == [5.0, 5.0], f"128-bit Float mismatch: {resf128}"

    print("ALL SCALABLE AND INFINITE WIDTHS TESTS PASSED!")

if __name__ == "__main__":
    test_scalable_and_infinite_widths()
