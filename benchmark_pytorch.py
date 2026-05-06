import time
import torch

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

for device in devices:
    print(f"\n--- {device.upper()} ---")
    for SIZE in [1024, 4096]:
        x = torch.randn(SIZE, SIZE, device=device)
        y = torch.randn(SIZE, SIZE, device=device)

        # warmup
        _ = torch.mm(x, y)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(10):
            z = torch.mm(x, y)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        avg_ms = (end - start) * 1000 / 10
        gflops = 2 * SIZE**3 / (avg_ms / 1000) / 1e9
        print(f"N={SIZE:5d}  avg={avg_ms:.2f} ms  {gflops:.1f} GFLOP/s")
