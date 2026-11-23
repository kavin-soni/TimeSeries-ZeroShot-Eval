"""Shared routines for the WLLFM 2026 paper analysis.

"The Complexity Router: Cost-Aware Foundation Model Deployment for Time Series
Forecasting" (Soni, Guduguntla, Das; WLLFM 2026 at IEEE BigData 2026).

Everything here uses only information available before deployment:
per-dataset FM and specialist choice, routing thresholds and the logistic
regression are all fit on the calibration split, and results are scored on the
holdout split.

Inputs (committed to the repo):
  results/per_series_results.csv  per-series MASE for every model
  results/series_features.csv     the four routing features per series

Threshold derivation, feature labels and the decile table are reused from
results/routing_analysis_holdout.py.
"""

import contextlib
import importlib.util
import io
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(REPO, 'results')

FMS = ['timesfm25', 'chronos']
SPECIALISTS = ['dlinear', 'patchtst']


def load_holdout_module(results_dir=RESULTS_DIR):
    """Import results/routing_analysis_holdout.py and point it at results_dir."""
    path = os.path.join(results_dir, 'routing_analysis_holdout.py')
    if not os.path.exists(path):
        path = os.path.join(RESULTS_DIR, 'routing_analysis_holdout.py')
    spec = importlib.util.spec_from_file_location('routing_analysis_holdout', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.FEATURES_PATH = os.path.join(results_dir, 'series_features.csv')
    mod.PER_SERIES_PATH = os.path.join(results_dir, 'per_series_results.csv')
    mod.FM_MODELS = set(FMS)
    mod.SPECIALIST_MODELS = set(SPECIALISTS)
    return mod


def quiet(fn, *args):
    """Call fn with its stdout suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def load_table(ra):
    """One row per series: dataset, series_id, the four features, MASE per model."""
    feats, per_series = quiet(ra.load_data)
    return quiet(ra.build_routing_table, feats, per_series)


def split(table, seed, datasets, cal_frac=0.4):
    """Random calibration/holdout split of series, stratified by dataset."""
    rng = np.random.default_rng(seed)
    cal_ids = {}
    for ds in datasets:
        ids = np.array(sorted(table[table.dataset == ds].series_id.unique()))
        cal_ids[ds] = set(rng.choice(ids, size=int(np.floor(cal_frac * len(ids))), replace=False))
    d = table[table.dataset.isin(datasets)]
    in_cal = np.array([s in cal_ids[ds] for ds, s in zip(d.dataset, d.series_id)])
    return d[in_cal].copy(), d[~in_cal].copy()


def pick_models(cal):
    """Per dataset, the FM and the specialist with the lowest calibration MASE."""
    picks = {}
    for ds in sorted(cal.dataset.unique()):
        c = cal[cal.dataset == ds]
        picks[ds] = (min(FMS, key=lambda m: c[m].mean()), min(SPECIALISTS, key=lambda m: c[m].mean()))
    return picks


def label(d, picks):
    """Attach the FM-class and specialist-class MASE given per-dataset picks."""
    d = d.copy()
    d['best_fm_mase'] = 0.0
    d['best_specialist_mase'] = 0.0
    for ds, (fm, sp) in picks.items():
        k = d.dataset == ds
        d.loc[k, 'best_fm_mase'] = d.loc[k, fm]
        d.loc[k, 'best_specialist_mase'] = d.loc[k, sp]
    d['fm_advantage'] = d.best_specialist_mase - d.best_fm_mase
    d['fm_wins'] = (d.fm_advantage > 0).astype(int)
    return d


def conditions(df, thr, features):
    """Boolean routing condition per feature, from derived thresholds."""
    out = {}
    for f in features:
        t = thr[f]
        x = df[f].values
        none = np.zeros(len(x), bool)
        if t['direction'] == '>=':
            out[f] = (x >= t['value']) if t.get('value') is not None else none
        elif t['direction'] == '<':
            out[f] = (x < t['value']) if t.get('value') is not None else none
        else:
            lo = (x < t['low_upper']) if t.get('low_upper') is not None else none
            hi = (x >= t['high_lower']) if t.get('high_lower') is not None else none
            out[f] = lo | hi
    return out


def oracle_curve(hol):
    """Holdout MASE when the top-k series by true FM advantage go to the FM (not deployable)."""
    d = hol.sort_values('fm_advantage', ascending=False).reset_index(drop=True)
    n = len(d)
    fm = d.best_fm_mase.values
    sp = d.best_specialist_mase.values
    cum_fm = np.concatenate([[0], np.cumsum(fm)])
    cum_sp = np.concatenate([[0], np.cumsum(sp[::-1])])[::-1]
    return (cum_fm + cum_sp) / n


def evaluate(ra, cal, hol):
    """Fit every router on cal and return its routing masks on hol.

    Masks: k1..k4 (k-of-4 vote), SE/CV/SAC/TR (single features), LR (logistic
    regression, p >= 0.5), LRm (logistic regression at the 1-of-4 FM share),
    SPEC (always specialist), FM (always FM).
    """
    F = ra.FEATURE_COLS
    picks = pick_models(cal)
    cal = label(cal, picks)
    hol = label(hol, picks)
    thr = quiet(ra.derive_thresholds, cal.copy())
    cc = conditions(cal, thr, F)
    ch = conditions(hol, thr, F)
    vc = sum(x.astype(int) for x in cc.values())
    vh = sum(x.astype(int) for x in ch.values())
    fm, sp = hol.best_fm_mase.values, hol.best_specialist_mase.values
    fmc, spc = cal.best_fm_mase.values, cal.best_specialist_mase.values

    M = {f'k{k}': vh >= k for k in (1, 2, 3, 4)}
    M.update(SE=ch['spectral_entropy'], CV=ch['cv'], SAC=ch['seasonal_autocorr'], TR=ch['trend_strength'])
    sc = StandardScaler().fit(cal[F])
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(cal[F]), cal.fm_wins)
    pr = lr.predict_proba(sc.transform(hol[F]))[:, 1]
    M['LR'] = pr >= 0.5
    n = len(hol)
    same_share = np.zeros(n, bool)
    same_share[np.argsort(-pr)[:int(round(M['k1'].mean() * n))]] = True
    M['LRm'] = same_share
    M['SPEC'] = np.zeros(n, bool)
    M['FM'] = np.ones(n, bool)

    cal_mase = {k: float(np.where(vc >= k, fmc, spc).mean()) for k in (1, 2, 3, 4)}
    return dict(M=M, fm=fm, sp=sp, hol=hol, cal=cal, thr=thr, picks=picks, pr=pr,
                kcal=min(cal_mase, key=cal_mase.get), cal_mase=cal_mase)


def delta(r, key):
    """Per-series MASE of a router minus always-FM."""
    return np.where(r['M'][key], r['fm'], r['sp']) - r['fm']


def bootstrap(x, n_boot=1000, seed=42):
    """Mean and 95% percentile interval of a paired per-series difference."""
    n = len(x)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    b = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
