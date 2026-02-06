# Data Sources

To replicate the results in this paper, please download the following datasets and place them in this folder using the naming convention specified in the root `README.md`.

### 1. M4 Dataset (Daily)
- **Source**: [M4 Competition Official Repository](https://github.com/Mcompetitions/M4-methods/tree/master/Dataset)
- **Files**: `Daily-train.csv`, `Daily-test.csv`, `M4-info.csv`

### 2. Traffic Dataset
- **Source**: [Informer/Autoformer Datasets](https://github.com/thuml/Autoformer) or [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
- **Description**: Contains hourly road occupancy rates from San Francisco Bay Area sensors.

### 3. Energy (ETTh1) Dataset
- **Source**: [ETDataset (Electricity Transformer Dataset)](https://github.com/zhouhaoyi/ETDataset)
- **File**: `ETTh1.csv`
- **Description**: 2 years of data from electricity transformers, recorded every hour.

### 4. Exchange Rate Dataset
- **Source**: [LSTNet Dataset Collection](https://github.com/laiguokun/LSTNet)
- **File**: `exchange_rate.csv`
- **Description**: Daily exchange rates of 8 foreign countries from 1990 to 2016.

## Preprocessing
If you are using the raw files from these sources, the `main_eval.py` script assumes they have been processed into the "Long" format (Series ID, Date, Value). You can use the preprocessing snippets located in the original notebook in the `notebooks/` directory to convert them.
