import numtoy

def test_numtoy_python():
    print("Initializing NumToy Engine...")
    engine = numtoy.Engine()

    print("Creating variables...")
    # Float(32) variables
    x = numtoy.Expr.var(engine, "x", [1.0, 2.0, 3.0], dtype="float", bits=32)
    y = numtoy.Expr.var(engine, "y", [4.0, 5.0, 6.0], dtype="float", bits=32)

    # Arithmetic: z = (x + y) * 2.5
    print("Building expression tree: z = (x + y) * 2.5")
    z = (x + y) * 2.5

    print("Compiling & Executing expression tree...")
    result_expr = z.execute(engine)

    print("Unpacking results...")
    results = result_expr.unpack(engine)
    print(f"Unpacked Results: {results}")

    # Expected: (1.0+4.0)*2.5 = 12.5, (2.0+5.0)*2.5 = 17.5, (3.0+6.0)*2.5 = 22.5
    expected = [12.5, 17.5, 22.5]
    print(f"Expected: {expected}")
    assert results == expected, f"Results mismatch! Got {results}, expected {expected}"
    print("Float32 test passed successfully!\n")

    # Int(16) test
    print("Testing packed Int16...")
    a = numtoy.Expr.var(engine, "a", [10, -20, 30], dtype="int", bits=16)
    b = numtoy.Expr.var(engine, "b", [5, 10, 15], dtype="int", bits=16)
    c = a * b
    c_res = c.execute(engine)
    c_vals = c_res.unpack(engine)
    print(f"Int16 Multiplication Results: {c_vals}")
    expected_c = [50.0, -200.0, 450.0]
    assert c_vals == expected_c, f"Int16 mismatch! Got {c_vals}, expected {expected_c}"
    print("Int16 test passed successfully!\n")

    # FloatingInt test (scale 2 for 10^-2 i.e. 100x multiplier)
    print("Testing FloatingInt (fixed point mapped to integer) with scale 2 (100x)...")
    fi1 = numtoy.Expr.var(engine, "fi1", [1.25, 2.5, 3.75], dtype="floating_int", scale=2)
    fi2 = numtoy.Expr.var(engine, "fi2", [0.75, 1.5, 2.25], dtype="floating_int", scale=2)
    fi_add = fi1 + fi2
    fi_res = fi_add.execute(engine)
    fi_vals = fi_res.unpack(engine)
    print(f"FloatingInt Addition Results: {fi_vals}")
    expected_fi = [2.0, 4.0, 6.0]
    assert fi_vals == expected_fi, f"FloatingInt mismatch! Got {fi_vals}, expected {expected_fi}"
    print("FloatingInt test passed successfully!\n")

    # Decorator compile test
    print("Testing @numtoy.compile decorator...")
    @numtoy.compile
    def my_kernel(x, y):
        return (x + y) * 3.0

    # Test via eager JIT execution through decorator
    x_list = [1.0, 2.0, 3.0]
    y_list = [4.0, 5.0, 6.0]
    res_expr = my_kernel(engine, x_list, y_list)
    res_vals = res_expr.unpack(engine)
    print(f"Decorator Result: {res_vals}")
    expected_dec = [15.0, 21.0, 27.0]
    assert res_vals == expected_dec, f"Decorator mismatch! Got {res_vals}, expected {expected_dec}"
    print("Decorator test passed successfully!\n")

    print("ALL PYTHON BINDINGS TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_numtoy_python()
