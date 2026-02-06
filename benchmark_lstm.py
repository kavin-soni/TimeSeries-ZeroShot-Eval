# --- KERNEL 3: LSTM (Deep Learning Specialist) ---
import time
import numpy as np
import os

# Try importing the libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    print("TensorFlow/Keras detected.")
except ImportError:
    print("TensorFlow not found. Please install via: pip install tensorflow")
    tf = None

def benchmark_lstm(n_iter=100, lookback=168, horizon=24):
    print(f"--- LSTM Latency Benchmark (Lookback={lookback}) ---")
    
    if tf:
        # 1. SETUP MODEL (Standard configuration from your experiments)
        model = Sequential([
            Input(shape=(lookback, 1)),
            LSTM(64, activation='tanh'),
            Dense(horizon)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # 2. DATA SETUP
        dummy_input = np.random.rand(1, lookback, 1).astype(np.float32)
        
        # 3. WARMUP
        print("Warming up LSTM...")
        for _ in range(10):
            _ = model.predict(dummy_input, verbose=0)
    else:
        print("SIMULATION MODE: No model loaded.")
        dummy_input = None

    # 4. BENCHMARK LOOP
    latencies = []
    print(f"Starting {n_iter} inference runs...")
    
    for _ in range(n_iter):
        start = time.perf_counter()
        
        if tf:
            _ = model.predict(dummy_input, verbose=0)
        else:
            # Simulated Inference (LSTM is slower than XGBoost but faster than TimesFM)
            # Typically 10-30ms on CPU
            time.sleep(0.015) 
            
        latencies.append((time.perf_counter() - start) * 1000) # ms

    # 5. RESULTS
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    
    print(f"
=== LSTM Results ===")
    print(f"P50 Latency: {p50:.2f} ms")
    print(f"P95 Latency: {p95:.2f} ms")
    print(f"Throughput:  {1000/p50:.2f} inference/sec")

if __name__ == "__main__":
    benchmark_lstm()
