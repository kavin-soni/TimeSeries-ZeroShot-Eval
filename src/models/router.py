import numpy as np
import pandas as pd
import argparse

def run_simulation(n_series=5000, threshold=0.15):
    """
    Simulates the 'Complexity Router' logic described in Section 6 of the paper.
    
    Generates a theoretical industrial fleet based on the Pareto Principle (80/20 rule):
    - 80% 'Stable' series (Class X): XGBoost performs comparably to Foundation Models.
    - 20% 'Complex' series (Class Z): Foundation Models provide significant lift (>15%).
    
    Args:
        n_series: Number of time series in the synthetic fleet.
        threshold: Accuracy improvement required to justify GPU routing (default 15%).
    """
    print(f"--- Running Complexity Router Simulation (N={n_series}) ---")
    np.random.seed(42) # Fixed seed for reproducibility
    
    # 1. Generate Synthetic Fleet (80/20 Split)
    n_stable = int(n_series * 0.80)
    n_complex = int(n_series * 0.20)
    
    # Generate MASE errors for Stable group (XGBoost ~= TimesFM)
    # Mean error ~0.6 with noise
    tfm_stable = np.random.uniform(0.4, 0.8, n_stable)
    xgb_stable = tfm_stable * np.random.uniform(0.95, 1.05, n_stable) 
    
    # Generate MASE errors for Complex group (TimesFM >> XGBoost)
    # XGBoost errors are 30-80% higher
    tfm_complex = np.random.uniform(0.5, 0.9, n_complex)
    xgb_complex = tfm_complex * np.random.uniform(1.3, 1.8, n_complex) 
    
    # Combine into fleet
    df = pd.DataFrame({
        'unique_id': np.arange(n_series),
        'MASE_TimesFM': np.concatenate([tfm_stable, tfm_complex]),
        'MASE_XGBoost': np.concatenate([xgb_stable, xgb_complex]),
        'Regime': ['Stable']*n_stable + ['Complex']*n_complex
    })
    
    # 2. Apply Router Logic (The Algorithm)
    # Rule: Route to TimesFM (GPU) ONLY if improvement > threshold
    df['Improvement'] = (df['MASE_XGBoost'] - df['MASE_TimesFM']) / df['MASE_XGBoost']
    
    df['Route_To'] = np.where(df['Improvement'] > threshold, 'TimesFM (GPU)', 'XGBoost (CPU)')
    
    # 3. Calculate Resource Impact (Volume)
    counts = df['Route_To'].value_counts()
    gpu_count = counts.get('TimesFM (GPU)', 0)
    cpu_count = counts.get('XGBoost (CPU)', 0)
    
    gpu_share = gpu_count / n_series
    cpu_share = cpu_count / n_series
    
    # 4. Calculate Accuracy Impact
    # Baseline: TimesFM for everyone (100% GPU)
    base_mase = df['MASE_TimesFM'].mean()
    # Hybrid: Error of the *selected* model
    df['System_MASE'] = np.where(df['Route_To'] == 'TimesFM (GPU)', df['MASE_TimesFM'], df['MASE_XGBoost'])
    hybrid_mase = df['System_MASE'].mean()

    # 5. Cost Analysis (New Section)
    # Derived from Table 2 Benchmarks: TimesFM=200ms, XGBoost=0.18ms
    cost_fm_ms = 200.0
    cost_xgb_ms = 0.18
    
    # Baseline Cost: All 5000 series go to FM
    total_cost_baseline = n_series * cost_fm_ms
    
    # Hybrid Cost: 
    # (GPU Routes * 200ms) + (CPU Routes * 0.18ms)
    total_cost_hybrid = (gpu_count * cost_fm_ms) + (cpu_count * cost_xgb_ms)
    
    # Savings Percentage
    pct_savings = (1 - (total_cost_hybrid / total_cost_baseline)) * 100
    
    # 6. Output Report
    print("
[Simulation Results]")
    print(f"Routing Threshold:      >{threshold*100}% Accuracy Gain required")
    print(f"Stable/Complex Split:   80% / 20%")
    print("-" * 30)
    print(f"XGBoost Routes (CPU):   {cpu_count} ({cpu_share*100:.1f}%)")
    print(f"TimesFM Routes (GPU):   {gpu_count} ({gpu_share*100:.1f}%)")
    print("-" * 30)
    print(f"Global GPU Savings:     {(1-gpu_share)*100:.1f}% (Volume Reduction)")
    print(f"Total Compute Cost Savings: {pct_savings:.2f}% (Weighted by Latency)")
    print("-" * 30)
    print(f"Accuracy Impact (MASE):")
    print(f"  Baseline (All GPU):   {base_mase:.4f}")
    print(f"  Hybrid System:        {hybrid_mase:.4f}")
    print(f"  Difference:           {hybrid_mase - base_mase:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Number of series to simulate")
    parser.add_argument("--threshold", type=float, default=0.15, help="Improvement threshold for routing")
    
    # For compatibility across environments
    import sys
    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    run_simulation(args.n, args.threshold)
