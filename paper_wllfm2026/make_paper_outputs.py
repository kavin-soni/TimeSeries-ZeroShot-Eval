"""Regenerate every number, table and data figure in the WLLFM 2026 paper.

Usage (from the repo root, CPU only, about 2 minutes):
    python paper_wllfm2026/make_paper_outputs.py
    python paper_wllfm2026/make_paper_outputs.py --results-dir results --out-dir paper_wllfm2026/outputs

Writes to the output directory:
    nums.tex                   LaTeX macros for every number quoted in the text
    tables/t2_tab.tex          Table III  (mean MASE per model)
    tables/m4_tab.tex          Table VI   (M4 Daily router results)
    tables/pooled_tab.tex      Table VII  (mixed Traffic + M4 portfolio)
    tables/cost_tab.tex        Table VIII (expected cost vs always-FM)
    pareto_m4.pdf              Fig. 2     (MASE vs FM share, M4 Daily)
    routing_features_m4.pdf    Fig. 3     (FM win rate by feature decile)
    summary.json               the same results in machine-readable form

Protocol: headline split seed 42 plus 100 repeated random splits (seeds
1000-1099), 40% calibration / 60% holdout. Per-dataset model choice,
thresholds and the logistic regression are refit on every calibration split.
Table IV (ETTh1/Exchange), Table V (throughput) and the regime map come from
separate runs; see README.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router_lib as rl  # noqa: E402

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})

HEADLINE_SEED = 42
REPEAT_SEEDS = range(1000, 1100)
# Router overhead in specialist calls: feature extraction ~0.9 ms on an
# M4-length history (time_features.py) / DLinear 0.13 ms per series (Table V).
EPS = 0.9 / 0.13

LABELS = {
    'SPEC': 'Always specialist (DLinear)', 'FM': 'Always FM',
    'k1': 'Router, 1-of-4 vote', 'k2': 'Router, 2-of-4 vote', 'k3': 'Router, 3-of-4 vote', 'k4': 'Router, 4-of-4 vote',
    'SE': 'Spectral entropy alone', 'CV': 'CV alone', 'SAC': 'Seasonal autocorr. alone', 'TR': 'Trend strength alone',
    'LR': 'Logistic regression ($p\\geq0.5$)', 'LRm': 'Logistic regression (same FM share as 1-of-4)',
}
KEYS = ['SPEC', 'FM', 'k1', 'k2', 'k3', 'k4', 'SE', 'CV', 'SAC', 'TR', 'LR', 'LRm']


def study(ra, table, name, datasets):
    """Headline split plus 100 repeated splits for one portfolio."""
    r = rl.evaluate(ra, *rl.split(table, HEADLINE_SEED, datasets))
    reps = [rl.evaluate(ra, *rl.split(table, s, datasets)) for s in REPEAT_SEEDS]
    S = {}
    for k in KEYS:
        d, lo, hi = rl.bootstrap(rl.delta(r, k))
        ds = np.array([rl.delta(x, k).mean() for x in reps])
        sh = np.array([x['M'][k].mean() for x in reps])
        S[k] = dict(share=float(r['M'][k].mean()), mase=float(np.where(r['M'][k], r['fm'], r['sp']).mean()),
                    d=d, lo=lo, hi=hi, rd=float(ds.mean()), rsd=float(ds.std()), frac=float((ds < 0).mean()),
                    rshare=float(sh.mean()), rshsd=float(sh.std()),
                    rp5=float(np.percentile(sh, 5)), rp95=float(np.percentile(sh, 95)))
    kcal = pd.Series([x['kcal'] for x in reps]).value_counts().to_dict()
    lr_vs_rule = np.array([(np.where(x['M']['LRm'], x['fm'], x['sp'])
                            - np.where(x['M']['k1'], x['fm'], x['sp'])).mean() for x in reps])
    hol, fm, sp = r['hol'], r['fm'], r['sp']
    n = len(hol)
    orc = rl.oracle_curve(hol)
    k1 = r['M']['k1']
    ex = dict(
        n=n, fmwin=float(hol.fm_wins.mean()), pureFM=float(fm.mean()), pureSpec=float(sp.mean()),
        fmspec=rl.bootstrap(sp - fm),
        rand=float((1 - k1.mean()) * sp.mean() + k1.mean() * fm.mean()),
        orc=float(orc[int(round(k1.mean() * n))]), orc_best=(float(orc.argmin() / n), float(orc.min())),
        prec=float(hol.fm_wins.values[k1].mean()), rest=float(hol.fm_wins.values[~k1].mean()),
        kcal=int(r['kcal']), kcalcounts={int(a): int(b) for a, b in kcal.items()},
        lrk1_diff=(float(lr_vs_rule.mean()), float(lr_vs_rule.std()), float((lr_vs_rule < 0).mean())),
        thr={f: {a: (None if b is None else (b if isinstance(b, str) else float(b)))
                 for a, b in t.items() if a in ('direction', 'value', 'low_upper', 'high_lower')}
             for f, t in r['thr'].items()},
        picks={a: list(b) for a, b in r['picks'].items()})
    if name == 'pooled':
        for d in ('traffic', 'm4'):
            kd = (hol.dataset == d).values
            ex['share_' + d] = {k: float(r['M'][k][kd].mean()) for k in ('k1', 'k2', 'LR')}
    grid = np.arange(0, n + 1, max(1, n // 200))
    order = np.argsort(-r['pr'])
    lr_curve = [float(np.where(np.isin(np.arange(n), order[:g]), fm, sp).mean()) for g in grid]
    return r, S, ex, dict(oracle=orc, lr=(grid / n, lr_curve))


def write_router_table(S, name, keys, path):
    head = ('\\textbf{Method} & \\textbf{FM share (\\%)} & \\textbf{MASE} & '
            '\\textbf{$\\Delta$ vs.\\ always-FM [95\\% CI]} & \\textbf{$\\Delta$, 100 splits} & \\textbf{Better (\\%)} \\\\')
    rules_after = {'m4': ('FM', 'k3', 'SAC'), 'pooled': ('FM', 'k4')}[name]
    rows = []
    for k in keys:
        s = S[k]
        if k == 'FM':
            rows.append(f"{LABELS[k]} & 100.0 & {s['mase']:.3f} & --- & --- & --- \\\\")
        else:
            rows.append(f"{LABELS[k]} & {100*s['share']:.1f} ({100*s['rshsd']:.1f}) & {s['mase']:.3f} & "
                        f"${s['d']:+.3f}$ [${s['lo']:+.3f}$, ${s['hi']:+.3f}$] & "
                        f"${s['rd']:+.3f}$ ({s['rsd']:.3f}) & {100*s['frac']:.0f} \\\\")
        if k in rules_after and k != keys[-1]:
            rows.append('\\midrule')
    with open(path, 'w') as fh:
        fh.write('\\begin{tabular}{lrrlrr}\n\\toprule\n' + head + '\n\\midrule\n'
                 + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def macros(res):
    """Every number quoted in the paper text, as LaTeX macros."""
    N = {}
    for name, P in (('m4', 'm'), ('pooled', 'p')):
        _, S, ex, _ = res[name]
        N[P + 'N'] = f"{ex['n']:,}".replace(',', '{,}')
        N[P + 'FmWin'] = f"{100*ex['fmwin']:.1f}"
        N[P + 'PureFM'] = f"{ex['pureFM']:.3f}"
        N[P + 'PureSpec'] = f"{ex['pureSpec']:.3f}"
        N[P + 'FmSpec'], N[P + 'FmSpecLo'], N[P + 'FmSpecHi'] = (f"{v:.3f}" for v in ex['fmspec'])
        N[P + 'Rand'] = f"{ex['rand']:.3f}"
        N[P + 'Orc'] = f"{ex['orc']:.3f}"
        N[P + 'OrcBestA'] = f"{ex['orc_best'][0]:.3f}"
        N[P + 'OrcBest'] = f"{ex['orc_best'][1]:.3f}"
        N[P + 'Prec'] = f"{100*ex['prec']:.1f}"
        N[P + 'Rest'] = f"{100*ex['rest']:.1f}"
        N[P + 'Gap'] = f"{100*(ex['rand']-S['k1']['mase'])/(ex['rand']-ex['orc']):.0f}"
        N[P + 'KcalOne'] = str(ex['kcalcounts'].get(1, 0))
        N[P + 'LrDiff'] = f"{ex['lrk1_diff'][0]:+.3f}"
        N[P + 'LrDiffSd'] = f"{ex['lrk1_diff'][1]:.3f}"
        N[P + 'LrBetter'] = f"{100*ex['lrk1_diff'][2]:.0f}"
        for k, kn in (('k1', 'One'), ('k2', 'Two'), ('k3', 'Three'), ('k4', 'Four'), ('SE', 'SE'), ('CV', 'CV'),
                      ('SAC', 'SAC'), ('TR', 'TR'), ('LR', 'LR'), ('LRm', 'LRm')):
            s, q = S[k], P + kn
            N[q + 'Share'] = f"{100*s['share']:.1f}"
            N[q + 'Saved'] = f"{100-100*s['share']:.1f}"
            N[q + 'Mase'] = f"{s['mase']:.3f}"
            N[q + 'D'] = f"{s['d']:+.3f}"
            N[q + 'Dabs'] = f"{abs(s['d']):.3f}"
            N[q + 'Lo'] = f"{s['lo']:+.3f}"
            N[q + 'Hi'] = f"{s['hi']:+.3f}"
            N[q + 'LoA'] = f"{abs(s['lo']):.3f}"
            N[q + 'HiA'] = f"{abs(s['hi']):.3f}"
            N[q + 'RD'] = f"{s['rd']:+.3f}"
            N[q + 'RDabs'] = f"{abs(s['rd']):.3f}"
            N[q + 'RSd'] = f"{s['rsd']:.3f}"
            N[q + 'Frac'] = f"{100*s['frac']:.0f}"
            N[q + 'RShare'] = f"{100*s['rshare']:.1f}"
            N[q + 'RShSd'] = f"{100*s['rshsd']:.1f}"
            N[q + 'RPlo'] = f"{100*s['rp5']:.0f}"
            N[q + 'RPhi'] = f"{100*s['rp95']:.0f}"
    ex = res['m4'][2]
    for f, nm in (('spectral_entropy', 'SEthr'), ('cv', 'CVthr'), ('seasonal_autocorr', 'SACthr')):
        t = ex['thr'][f]
        N['m' + nm] = f"{(t.get('value') if t.get('value') is not None else t.get('high_lower')):.3f}"
    ep = res['pooled'][2]
    for d in ('traffic', 'm4'):
        for k in ('k1', 'k2', 'LR'):
            N['p' + d.capitalize() + k.upper() + 'Sh'] = f"{100*ep['share_' + d][k]:.1f}"
    return N


def cost(res, N, path):
    """Cost model E[C] = a*C_FM + (1-a)*C_spec + eps, as a fraction of always-FM (Table VIII)."""
    S = res['m4'][1]
    a1, aL = S['k1']['rshare'], S['LR']['rshare']
    N['costEps'] = f"{EPS:.0f}"
    N['costBE'] = f"{1 + EPS / (1 - a1):.0f}"
    N['mOneRShareShort'] = f"{100*a1:.0f}"
    rows = []
    for R in (10, 100, 1000):
        f = lambda a, e: (a * R + (1 - a) + e) / R  # noqa: E731
        rows.append(f"{R:,}$\\times$ & {f(a1,0):.2f} & {f(a1,EPS):.2f} & {f(aL,0):.2f} & {f(aL,EPS):.2f} \\\\"
                    .replace(',', '{,}'))
        N[f'costOne{R}'] = f"{f(a1, EPS):.2f}"
        N[f'costOne{R}Z'] = f"{f(a1, 0):.2f}"
        N[f'costLR{R}'] = f"{f(aL, EPS):.2f}"
    with open(path, 'w') as fh:
        fh.write('\\begin{tabular}{lrrrr}\n\\toprule\n & \\multicolumn{2}{c}{\\textbf{Router, 1-of-4}} & '
                 '\\multicolumn{2}{c}{\\textbf{Log.\\ reg.\\ ($p\\geq0.5$)}} \\\\\n'
                 '\\textbf{$C_{\\text{FM}}/C_{\\text{spec}}$} & \\textbf{w/o} & \\textbf{with} & \\textbf{w/o} & '
                 '\\textbf{with} \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def macro_name(key):
    digits = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four',
              '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'}
    return ''.join(digits.get(c, c) for c in key)


def model_accuracy_table(table, path):
    """Table III: mean MASE over all series per model (Traffic, M4 Daily)."""
    rows = []
    means = {ds: {m: table[table.dataset == ds][m].mean() for m in rl.FMS + rl.SPECIALISTS} for ds in ('traffic', 'm4')}
    best = {ds: min(means[ds], key=means[ds].get) for ds in means}
    names = {'timesfm25': ('TimesFM 2.5', 'FM'), 'chronos': ('Chronos', 'FM'),
             'patchtst': ('PatchTST', 'Spec.'), 'dlinear': ('DLinear', 'Spec.')}
    for m in ('timesfm25', 'chronos', 'patchtst', 'dlinear'):
        cells = [f"\\textbf{{{means[ds][m]:.3f}}}" if best[ds] == m else f"{means[ds][m]:.3f}" for ds in ('traffic', 'm4')]
        rows.append(f"{names[m][0]} & {names[m][1]} & {cells[0]} & {cells[1]} \\\\")
    with open(path, 'w') as fh:
        fh.write('\\begin{tabular}{llcc}\n\\toprule\n\\textbf{Model} & \\textbf{Class} & \\textbf{Traffic} & '
                 '\\textbf{M4 Daily} \\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
    return {ds: {m: round(float(v), 3) for m, v in means[ds].items()} for ds in means}


def text_numbers(ra, res):
    """Numbers quoted in the text that are not macros."""
    r = res['m4'][0]
    hol = r['hol']
    # "Why not select per series": per-series better of the two FMs by realized
    # test error (hindsight, not deployable) vs the calibration-selected FM.
    hindsight = float(hol[rl.FMS].min(axis=1).mean())
    trend = ra._decile_table(r['cal'].copy(), 'trend_strength')
    return dict(m4_holdout_fm_selected=round(float(r['fm'].mean()), 3),
                m4_holdout_fm_hindsight_min=round(hindsight, 3),
                m4_cal_trend_top_decile_fm_win_rate_pct=round(100 * float(trend['win_rate'].iloc[-1]), 1))


def pareto_figure(res, path):
    """Fig. 2: holdout MASE against share routed to the FM (M4 Daily, headline split)."""
    _, S, ex, cv = res['m4']
    n = ex['n']
    with plt.rc_context({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': 0.6}):
        fig, ax = plt.subplots(figsize=(3.45, 3.35))
        ax.plot(np.arange(n + 1) / n * 100, cv['oracle'], color='#555555', lw=1.4, label='Oracle ranking (not deployable)')
        ax.plot(cv['lr'][0] * 100, cv['lr'][1], color='#D55E00', lw=1.2, label='Logistic regression, cutoff swept')
        ax.plot([0, 100], [ex['pureSpec'], ex['pureFM']], ls='--', color='#999999', lw=1, label='Random assignment')
        for k, lb, mk, c in (('SPEC', 'Always specialist', 's', '#000000'), ('FM', 'Always FM', 'D', '#000000'),
                             ('k1', 'Router, 1-of-4', 'o', '#0072B2'), ('k2', 'Router, 2-of-4', 'o', '#56B4E9'),
                             ('k3', 'Router, 3-of-4', 'o', '#CC79A7'),
                             ('LR', 'Logistic regression, $p\\geq0.5$', '*', '#D55E00')):
            ax.scatter(S[k]['share'] * 100, S[k]['mase'], marker=mk, s=34 if mk != '*' else 60, color=c,
                       edgecolor='black', linewidth=0.5, zorder=5, label=lb)
        ax.set_xlabel('Series routed to the FM (%)')
        ax.set_ylabel('MASE (M4 Daily holdout)')
        ax.set_xlim(-3, 103)
        ax.grid(alpha=0.25, lw=0.4)
        # Top axis: relative cost at an assumed FM/specialist ratio of 1000, no router overhead.
        top = ax.secondary_xaxis('top', functions=(lambda s: 1 + s * 9.99, lambda c: (c - 1) / 9.99))
        top.set_xlabel('Relative inference cost (specialist = 1)', fontsize=7)
        top.set_xticks([1, 250, 500, 750, 1000])
        top.tick_params(labelsize=6.5)
        ax.legend(fontsize=6.0, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False,
                  columnspacing=1.0, labelspacing=0.3)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def decile_figure(ra, res, path):
    """Fig. 3: FM win rate by decile of each feature on the M4 calibration set (headline split)."""
    r = res['m4'][0]
    cal, thr = r['cal'], r['thr']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, f in zip(axes.flatten(), ra.FEATURE_COLS):
        g = ra._decile_table(cal.copy(), f)
        ax.bar(g['_decile'], g['win_rate'] * 100, color='steelblue', alpha=0.75, edgecolor='white')
        ax.axhline(60, color='red', ls='--', lw=1.2, label='60% bar')
        ax.set_xlabel(f"Decile of {ra.FEATURE_LABELS[f]}", fontsize=9)
        ax.set_ylabel('FM win rate (%)', fontsize=9)
        ax.set_title(ra.FEATURE_LABELS[f], fontsize=10, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_xticks(g['_decile'])
        for _, row in g.iterrows():
            ax.text(row['_decile'], row['win_rate'] * 100 + 1.5, f"n={int(row['n'])}", ha='center', va='bottom', fontsize=7)
        t = thr[f]
        if f in ('spectral_entropy', 'cv'):
            ax.axvline(t['decile'], color='orange', ls=':', lw=1.5,
                       label=f"cut at decile {t['decile']}: feature $\\geq$ {t['value']:.4f}")
        elif f == 'seasonal_autocorr':
            ax.axvline(t['high_decile'], color='orange', ls=':', lw=1.5,
                       label=f"cut at decile {t['high_decile']}: feature $\\geq$ {t['high_lower']:.4f}")
        else:
            ax.plot([], [], ' ', label='no condition (low end never reaches 60%)')
        ax.legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default=rl.RESULTS_DIR, help='folder with the two input CSVs')
    ap.add_argument('--out-dir', default=os.path.join(rl.HERE, 'outputs'), help='where to write outputs')
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, 'tables'), exist_ok=True)

    ra = rl.load_holdout_module(args.results_dir)
    table = rl.load_table(ra)
    print('Running headline split and 100 repeated splits (M4 Daily, then mixed portfolio)...', flush=True)
    res = {name: study(ra, table, name, ds) for name, ds in (('m4', ['m4']), ('pooled', ['traffic', 'm4']))}

    write_router_table(res['m4'][1], 'm4', ['SPEC', 'FM', 'k1', 'k2', 'k3', 'SE', 'CV', 'SAC', 'LR', 'LRm'],
                       os.path.join(args.out_dir, 'tables', 'm4_tab.tex'))
    write_router_table(res['pooled'][1], 'pooled', ['SPEC', 'FM', 'k1', 'k2', 'k3', 'k4', 'LR', 'LRm'],
                       os.path.join(args.out_dir, 'tables', 'pooled_tab.tex'))
    N = macros(res)
    cost(res, N, os.path.join(args.out_dir, 'tables', 'cost_tab.tex'))
    with open(os.path.join(args.out_dir, 'nums.tex'), 'w') as fh:
        fh.write('\n'.join(f"\\newcommand{{\\{macro_name(k)}}}{{{v}}}" for k, v in N.items()) + '\n')
    accuracy = model_accuracy_table(table, os.path.join(args.out_dir, 'tables', 't2_tab.tex'))
    text = text_numbers(ra, res)
    pareto_figure(res, os.path.join(args.out_dir, 'pareto_m4.pdf'))
    decile_figure(ra, res, os.path.join(args.out_dir, 'routing_features_m4.pdf'))
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as fh:
        json.dump({'macros': N, 'mean_mase': accuracy, 'text_numbers': text,
                   'm4': res['m4'][2], 'pooled': res['pooled'][2],
                   'm4_methods': res['m4'][1], 'pooled_methods': res['pooled'][1]}, fh, indent=1, default=str)

    S, ex = res['m4'][1], res['m4'][2]
    print(f"M4 Daily holdout (n={ex['n']}): always-FM {ex['pureFM']:.3f}, 1-of-4 vote {S['k1']['mase']:.3f} "
          f"with {100*S['k1']['share']:.1f}% to FM; beats always-FM in {100*S['k1']['frac']:.0f}% of 100 splits; "
          f"break-even cost ratio ~{N['costBE']}.")
    print(f"Outputs written to {args.out_dir}")


if __name__ == '__main__':
    main()
