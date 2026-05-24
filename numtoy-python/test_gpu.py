import numtoy

def test_gpu_execution():
    print("Initializing NumToy Engine...")
    engine = numtoy.Engine()

    print("Creating variables...")
    x = numtoy.Expr.var(engine, "x", [1.0, 2.0, 3.0], dtype="float", bits=32)
    y = numtoy.Expr.var(engine, "y", [4.0, 5.0, 6.0], dtype="float", bits=32)

    print("Building expression tree: z = (x + y) * 2.5")
    z = (x + y) * 2.5

    print("Executing on GPU...")
    # device_str: "gpu"
    result_expr = z.execute(engine, device="gpu")

    print("Unpacking results...")
    results = result_expr.unpack(engine)
    print(f"Unpacked Results: {results}")

    expected = [12.5, 17.5, 22.5]
    print(f"Expected: {expected}")
    assert results == expected, f"Results mismatch! Got {results}, expected {expected}"
    print("Float32 GPU test passed successfully!\n")

if __name__ == "__main__":
    test_gpu_execution()
