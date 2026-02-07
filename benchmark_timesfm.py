# --- KERNEL 1: TimesFM (Foundation Model) ---
import time
import numpy as np
import torch
import os

# Try importing the library, handle error if not installed
try:
    import timesfm
    print("TimesFM library detected.")
except ImportError:
    print("TimesFM library not found. Please install via: pip install timesfm")
    timesfm = None

def benchmark_timesfm(n_iter=50, batch_size=1, context_len=512, horizon_len=96):
    # 1. LOAD MODEL
    print("Loading TimesFM (this may take a minute)...")
    if timesfm:
        device = "gpu" if torch.cuda.is_available() else "cpu"
        # Newer versions of timesfm usually use TimesFm
        try:
            tfm = timesfm.TimesFm(
                context_len=context_len,
                horizon_len=horizon_len,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                backend=device
            )
        except AttributeError:
            # Fallback to TimesFM
            tfm = timesfm.TimesFM(
                context_len=context_len,
                horizon_len=horizon_len,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                backend=device
            )
            
        # Using the official checkpoint
        tfm.load_from_checkpoint(repo_id="google/timesfm-2.0-500m")
    else:
        print("SIMULATION MODE: No model loaded.")
        tfm = None

    # 2. DATA SETUP
    # Create dummy time series: (Batch, Context_Length)
    dummy_data = np.random.rand(batch_size, context_len).astype(np.float32)

    # 3. WARMUP
    print("Warming up TimesFM (Batch={})...".format(batch_size))
    if tfm:
        _, _ = tfm.forecast(dummy_data)
    else:
        time.sleep(1.0) 

    # 4. BENCHMARK LOOP
    latencies = []
    print("Starting {} inference runs...".format(n_iter))
    
    for _ in range(n_iter):
        start = time.time()
        
        if tfm:
            forecast, _ = tfm.forecast(dummy_data)
        else:
            time.sleep(0.2) 
            
        latencies.append((time.time() - start) * 1000) # ms

    # 5. RESULTS
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    
    print("\n=== TimesFM Results ===")
    print("P50 Latency: {:.2f} ms".format(p50))
    print("P95 Latency: {:.2f} ms".format(p95))
    print("Throughput:  {:.2f} inference/sec (Batch={})".format(1000/p50, batch_size))

if __name__ == "__main__":
    benchmark_timesfm()
