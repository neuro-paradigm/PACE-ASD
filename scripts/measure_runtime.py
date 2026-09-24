"""
Measure actual inference runtime and memory for PACE-ASD A1 vs A2.
Saves reproducible results to results/runtime_measurements.json.
"""
import os
import sys
import time
import json
import platform
import numpy as np
import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from model import ASDMotionModel

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def benchmark_model(model, x, device, n_warmup=20, n_runs=100):
    model.eval()
    model.to(device)
    x = x.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0) # in ms

    peak_mem_mb = None
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "peak_gpu_mem_mb": peak_mem_mb,
        "n_runs": n_runs,
    }

def main():
    has_cuda = torch.cuda.is_available()
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda") if has_cuda else None

    # Load A1 and A2 models
    ckpt_a1 = "models/A1/fold1_seed42.pt"
    ckpt_a2 = "models/A2/fold1_seed42.pt"

    # Input tensor shape: (batch_size=1, T=300, V=33, C=2)
    # Testing with typical clip length: 124 frames (median) and 300 frames (max)
    x_300 = torch.randn(1, 300, 33, 2, dtype=torch.float32)

    results = {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": has_cuda,
            "cuda_device_name": torch.cuda.get_device_name(0) if has_cuda else None,
            "torch_cuda_version": torch.version.cuda if has_cuda else None,
        },
        "models": {}
    }

    print("Running PACE-ASD Runtime Benchmark...")
    print(f"System: {results['environment']['platform']} | CPU: {results['environment']['processor']}")
    print(f"PyTorch: {torch.__version__} | CUDA available: {has_cuda}")

    for model_name, ckpt_path in [("A1", ckpt_a1), ("A2", ckpt_a2)]:
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found.")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("config", {
            "model": {
                "spatial_dim": 128,
                "conv1d_channels": 32,
                "dropout": 0.4,
                "event_block_size": 15,
                "event_top_m": 8,
                "transformer_heads": 4,
                "transformer_layers": 1,
            }
        })
        if "model" not in cfg:
            cfg = {"model": cfg}
            
        use_gate = True if model_name == "A1" else False
        model = ASDMotionModel(
            cfg,
            use_gate=use_gate,
            use_transformer=True,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        tot_p, train_p = count_parameters(model)
        print(f"\n{model_name}: Total Parameters = {tot_p:,}, Trainable = {train_p:,}")

        # Benchmark CPU
        bench_cpu = benchmark_model(model, x_300, device_cpu)
        print(f"  CPU ({bench_cpu['n_runs']} runs): {bench_cpu['mean_ms']:.2f} +- {bench_cpu['std_ms']:.2f} ms per clip (median: {bench_cpu['median_ms']:.2f} ms)")

        bench_gpu = None
        if has_cuda and device_gpu is not None:
            bench_gpu = benchmark_model(model, x_300, device_gpu)
            print(f"  GPU ({bench_gpu['n_runs']} runs): {bench_gpu['mean_ms']:.2f} +- {bench_gpu['std_ms']:.2f} ms per clip (peak mem: {bench_gpu['peak_gpu_mem_mb']:.2f} MB)")

        results["models"][model_name] = {
            "total_parameters": tot_p,
            "trainable_parameters": train_p,
            "checkpoint": ckpt_path,
            "cpu_benchmark": bench_cpu,
            "gpu_benchmark": bench_gpu,
        }

    os.makedirs("results", exist_ok=True)
    out_file = "results/runtime_measurements.json"
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nBenchmark completed. Results written to {out_file}.")

if __name__ == "__main__":
    main()
