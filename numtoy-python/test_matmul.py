import numtoy

def test_dequantize_matmul():
    print("Initializing NumToy Engine...")
    engine = numtoy.Engine()

    print("Creating variables...")
    # Activations: [1, 8] - fp32
    act_vals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    activations = numtoy.Tensor.from_values(engine, act_vals, [1, 8], dtype="float", bits=32)
    
    # Weights: [8, 1] - custom u3 (ScalableInt with 3 bits, or just int 3)
    # We will use int with 3 bits for testing
    w_vals = [1, 2, 3, 4, -1, -2, -3, -4]
    weights = numtoy.Tensor.from_values(engine, w_vals, [8, 1], dtype="int", bits=3)

    print("Building DequantizeMatmul...")
    z = numtoy.Tensor.dequantize_matmul(activations, weights)

    print("Executing on CPU...")
    result_tensor = z.execute(engine, device="cpu")

    print("Unpacking results...")
    results = result_tensor.numpy(engine)
    print(f"Unpacked Results: {results}")

    # Expected: 1*1 + 1*2 + 1*3 + 1*4 + 1*-1 + 1*-2 + 1*-3 + 1*-4 = 0
    # Wait, actually -1 in 3-bit might wrap, but assuming it uses proper sign extension.
    # Let's see what the system produces first.

if __name__ == "__main__":
    test_dequantize_matmul()
