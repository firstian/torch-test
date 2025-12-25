import time 
import torch

print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"PyTorch's CUDA version: {torch.version.cuda}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

def benchmark_gemm(size=10000, iterations=100):
    # Initialize large matrices on the GPU
    a = torch.randn(size, size, device='cuda', dtype=torch.float16)
    b = torch.randn(size, size, device='cuda', dtype=torch.float16)

    # Warm up (JIT/Driver initialization)
    print("Warming up...")
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    # Timing
    print(f"Running {iterations} iterations of {size}x{size} matrix multiplication...")
    start_time = time.time()
    for _ in range(iterations):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    end_time = time.time()

    duration = end_time - start_time
    tflops = (2 * size**3 * iterations) / duration / 1e12
    print(f"Avg Duration: {duration/iterations:.4f}s")
    print(f"Performance: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_gemm()
    else:
        print("CUDA NOT AVAILABLE - Check your drivers!")
