# Zero-Shot Time Series Foundation Model Evaluation

This repository provides the code and experimental framework for the paper: **"An Applied Evaluation of Decoder-Only Foundation Models for Zero-Shot Time Series Forecasting"**.

## Abstract
Time series forecasting remains central to operational decision-making across domains such as transportation, energy systems, and financial planning. This work presents an applied evaluation of a decoder-only foundation model (TimesFM) in a strictly zero-shot setting, comparing it against commonly deployed supervised approaches (XGBoost, LSTM) and statistical baselines (Seasonal Naive). We analyze performance across four operational regimes: periodic human-centric systems (traffic), physically constrained processes (energy), stochastic financial markets, and heterogeneous demand forecasting (M4).

## Repository Structure

```
.
├── data/               # Data preparation instructions
├── notebooks/          # Original exploration notebooks
├── results/            # Benchmark metrics (CSV and Markdown)
└── src/
    ├── models/         # TimesFM wrapper and Baseline implementations
    └── utils/          # Data loaders and metric utilities
├── main_eval.py        # Entry point for running benchmarks
├── requirements.txt    # Python dependencies
└── README.md
```

## Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Place your dataset files in the `data/` directory. The scripts expect the following naming convention:
- `traffic_train.csv`, `traffic_test.csv`
- `etth1_train.csv`, `etth1_test.csv`
- `exchange_train.csv`, `exchange_test.csv`
- `m4_train.csv`, `m4_test.csv`

### 3. Running Benchmarks
To run the evaluation for a specific operational regime (e.g., Traffic):
```bash
python main_eval.py --dataset traffic
```

## Final Benchmark Results

| Dataset | Metric | TimesFM (Zero-Shot) | SNAIVE | XGBoost | LSTM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M4 (n=4227)** | MASE | 1.426 | 3.278 | 1.763 | 3.735 |
| | sMAPE | 4.15% | 3.05% | 1.39% | 3.35% |
| | RMSE | 290.37 | 212.00 | 125.49 | 232.91 |
| **Traffic (n=862)** | MASE | 0.482 | 1.096 | 0.514 | 0.861 |
| | sMAPE | 19.43% | 28.66% | 15.21% | 27.32% |
| | RMSE | 0.019 | 0.033 | 0.017 | 0.021 |
| **Energy (n=7)** | MASE | 2.576 | 2.092 | 0.573 | 0.836 |
| | sMAPE | 56.36% | 68.66% | 25.85% | 32.18% |
| | RMSE | 2.887 | 3.940 | 1.187 | 1.700 |
| **Exchange (n=8)** | MASE | 12.374 | 13.861 | 3.942 | 9.677 |
| | sMAPE | 3.76% | 12.10% | 0.81% | 2.77% |
| | RMSE | 0.034 | 0.093 | 0.007 | 0.022 |

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
