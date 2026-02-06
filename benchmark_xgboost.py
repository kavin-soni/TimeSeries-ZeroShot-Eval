# --- KERNEL 2: XGBoost (Supervised Baseline) ---
import time
import numpy as np
import os

# Try importing the library
try:
    import xgboost as xgb
    print("XGBoost library detected.")
except ImportError:
    print("XGBoost library not found. Please install via: pip install xgboost")
    xgb = None

def benchmark_xgboost(n_iter=100, n_features=15):
    print("Setting up XGBoost Benchmark (CPU)...")
    
    # 1. CREATE MOCK DATA & MODEL
    # Standard feature count for your Traffic/M4 setups
    X_dummy = np.random.rand(1000, n_features).astype(np.float32)
    y_dummy = np.random.rand(1000)

    if xgb:
        # Train a small model to simulate real usage
        dtrain = xgb.DMatrix(X_dummy, label=y_dummy)
        params = {'objective': 'reg:squarederror', 'nthread': 1} # Forcing single thread to test raw CPU speed
        bst = xgb.train(params, dtrain, num_boost_round=100)
    else:
        print("SIMULATION MODE: No model loaded.")
        bst = None

    # Single inference point (Batch size 1)
    test_point = xgb.DMatrix(X_dummy[:1, :]) if xgb else None

    # 2. WARMUP
    print("Warming up XGBoost...")
    if bst:
        _ = bst.predict(test_point)
    else:
        time.sleep(0.5)

    # 3. BENCHMARK LOOP
    latencies = []
    print("Starting {} inference runs...".format(n_iter))
    
    for _ in range(n_iter):
        start = time.time()
        
        if bst:
            _ = bst.predict(test_point)
        else:
            # Simulated Inference (XGBoost is very fast on CPU, approx 1-5ms)
            time.sleep(0.002) 
            
        latencies.append((time.time() - start) * 1000) # ms

    # 4. RESULTS
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    
    print("
=== XGBoost Results (CPU) ===")
    print("P50 Latency: {:.2f} ms".format(p50))
    print("P95 Latency: {:.2f} ms".format(p95))
    print("Throughput:  {:.2f} inference/sec".format(1000/p50))

if __name__ == "__main__":
    benchmark_xgboost()
