# The Complexity Router: Cost-Aware Foundation Model Deployment for Time Series Forecasting

**Kavin Soni, Vamshi Guduguntla, Debanshu Das** (Google)

Accepted at the **Fourth Workshop on Large Language and Foundation Models (WLLFM 2026)**, held with **IEEE BigData 2026**, Phoenix, AZ, December 2026. IEEE Xplore link to be added on publication.

[arXiv](https://arxiv.org/abs/2605.24381) | [Reproduce the paper](#reproduce-the-paper-cpu-about-1-minute) | [Earlier version (arXiv v1)](#earlier-version-arxiv-v1)

## Overview

Time series foundation models (FMs) forecast zero-shot, but they cost far more to run than lightweight supervised specialists. The Complexity Router decides **before any model runs** whether each series goes to an FM or to a specialist, using four features of its history: spectral entropy, coefficient of variation, seasonal autocorrelation and trend strength.

The evaluation uses only information available before deployment. The FM and the specialist are chosen per dataset on a calibration split, routing thresholds and a logistic regression are fit on the same split, and results are reported on held-out series over a headline split and 100 repeated random splits.

## Key results

M4 Daily holdout (n = 2,537), where the FM (Chronos) and the specialist (DLinear) are close in accuracy:

| Method | Series to FM | MASE | vs. always-FM |
|---|---|---|---|
| Always FM | 100% | 1.206 | |
| Always specialist | 0% | 1.233 | +0.027 |
| **1-of-4 vote rule** | **31.6%** | **1.179** | **-0.027**, better in all 100 repeated routing splits |
| Logistic regression (same FM share) | 31.6% | 1.179 | -0.027 |
| Oracle ranking (not deployable) | 50.3% | 1.059 | |

- On a mixed Traffic and M4 portfolio, the logistic regression beats the vote rule at equal FM share in 93% of splits.
- Counting the router's own overhead, routing saves cost once an FM call costs more than about **11 specialist calls**.
- Per-series hindsight selection (the better of two FMs by realized test error) would lower the M4 FM-class MASE from 1.206 to 1.065. It is not deployable, so the paper does not use it.

See the paper's Limitations section for scope: within-dataset routing evidence comes from M4 Daily, each model was trained once, and the cost ratio is a parameter.

## Reproduce the paper (CPU, about 1 minute)

Every number, table and data figure in the paper is regenerated from the two committed per-series result files, without rerunning any model.

```bash
pip install -r paper_wllfm2026/requirements.txt
python paper_wllfm2026/make_paper_outputs.py
```

Outputs go to `paper_wllfm2026/outputs/`:

| Output | Paper item |
|---|---|
| `tables/t2_tab.tex` | Table III, mean MASE per model |
| `tables/m4_tab.tex` | Table VI, M4 Daily router results |
| `tables/pooled_tab.tex` | Table VII, mixed Traffic and M4 portfolio |
| `tables/cost_tab.tex` | Table VIII, expected cost vs. always-FM |
| `pareto_m4.pdf` | Fig. 2, MASE vs. share routed to the FM |
| `routing_features_m4.pdf` | Fig. 3, FM win rate by feature decile |
| `nums.tex` | LaTeX macros for every number quoted in the text |
| `summary.json` | All of the above in machine-readable form |

Other paper items:

```bash
python paper_wllfm2026/make_regime_map.py   # Fig. 1, regime map
python paper_wllfm2026/time_features.py     # router overhead (feature-extraction latency)
```

- Table IV (ETTh1 and Exchange) comes from separate, unseeded runs of the model scripts below; the paper reports ranges where runs differ.
- Table V (throughput) was measured separately on GPU (T4) and CPU (Xeon) hardware.

Inputs, in `results/`:

| File | Contents |
|---|---|
| `per_series_results.csv` | MASE, RMSE and sMAPE per model, dataset and series |
| `series_features.csv` | The four routing features and history length per series |

## Rerun the models (GPU for FMs)

### Datasets

| Dataset | Domain | Freq | Series | Source |
|---|---|---|---|---|
| M4 Daily | Demand | Daily | 4,227 | [M4 competition](https://github.com/Mcompetitions/M4-methods/tree/master/Dataset) |
| Traffic (PeMS) | Transport | Hourly | 862 | [PeMS](https://pems.dot.ca.gov/), preprocessed in [Time-Series-Library](https://github.com/thuml/Time-Series-Library) |
| ETTh1 | Energy | Hourly | 7 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| Exchange Rate | Finance | Daily | 8 | [LSTNet data](https://github.com/laiguokun/multivariate-time-series-data) |

Place the raw files (`traffic.csv`, `ETTh1.csv`, `exchange_rate.csv`, `Daily-train.csv`, `Daily-test.csv`, `M4-info.csv`) in the repo root and run:

```bash
python data_preprocessing.py
```

### Models used in the paper

```bash
pip install pandas numpy scipy scikit-learn matplotlib torch
python run_dlinear_all.py
python run_patchtst_all.py

# Foundation models (GPU recommended)
git clone https://github.com/google-research/timesfm.git && pip install -e timesfm
pip install chronos-forecasting
python run_foundation_models.py --model timesfm25 --dataset m4
python run_foundation_models.py --model chronos   --dataset m4
python run_foundation_models.py --model timesfm25 --dataset traffic
python run_foundation_models.py --model chronos   --dataset traffic

python results/extract_series_features.py
```

Specialist training and Chronos sampling are not seeded, so reruns differ from the committed results (the paper reports about 0.05 MASE between runs of DLinear on Traffic). The committed CSVs are the runs behind the paper's numbers.

## Earlier version (arXiv v1)

arXiv v1 ("Assessing the Operational Viability of Foundation Models for Time Series Forecasting") labeled series using realized test error, which is not available before deployment; its router results are superseded by the WLLFM 2026 paper. Kept for reference only:

- `results/routing_analysis.py`, `results/routing_analysis_holdout.py`: v1 analyses (the paper code imports the latter only for threshold derivation and decile tables).
- `run_xgboost.py`, `run_lstm.py`, TimesFM 2.0 in `run_foundation_models.py`: v1-only models. "The paper" and "Table 2" in these scripts refer to v1.

## Repository structure

```
paper_wllfm2026/
  router_lib.py              # splits, model choice, vote rule, logistic regression, oracle, bootstrap
  make_paper_outputs.py      # all tables, Figs. 2-3 and nums.tex
  make_regime_map.py         # Fig. 1
  time_features.py           # router overhead timing
  requirements.txt
results/
  per_series_results.csv     # per-series MASE for every model (paper inputs)
  series_features.csv        # routing features per series (paper inputs)
  extract_series_features.py
  routing_analysis_holdout.py
  routing_analysis.py        # v1 only
src/                         # DLinear, PatchTST and metric implementations
run_dlinear_all.py, run_patchtst_all.py, run_foundation_models.py
run_xgboost.py, run_lstm.py  # v1 only
data_preprocessing.py
```

## Citation

```bibtex
@inproceedings{soni2026complexity,
  title         = {The Complexity Router: Cost-Aware Foundation Model Deployment for Time Series Forecasting},
  author        = {Soni, Kavin and Guduguntla, Vamshi and Das, Debanshu},
  booktitle     = {Fourth Workshop on Large Language and Foundation Models (WLLFM), IEEE International Conference on Big Data},
  year          = {2026},
  eprint        = {2605.24381},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
