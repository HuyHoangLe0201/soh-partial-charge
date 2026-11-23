import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

PRIMARY = 'primary'
KINDS = ['primary', 'isotropic_map', 'target_secant', 'endpoint_secant', 'source_logridge',
         'scalar_map', 'carry_last', 'cycle_trend', 'covariance_exact', 'best_segment']
SCENARIOS = ['clean', 'vp5', 'vm5', 'vp10', 'vm10']
STRESS = SCENARIOS[1:]
STAGES = ['loco', 'development', 'confirmation']

_SRC_DIR = Path(__file__).resolve().parent
P = _SRC_DIR.parent if _SRC_DIR.name == 'src' else _SRC_DIR
OUT = P / 'analysis_stage2'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RECEIPTS = {}

def load_stage(stage):
    rows = []
    mets = []
    d = P / 'stage2' / stage
    rp = d / f'{stage}_row_predictions.csv'
    cm = d / f'{stage}_cell_metrics.csv'
    if not rp.exists():
        raise FileNotFoundError(rp)
    if not cm.exists():
        raise FileNotFoundError(cm)
    r = pd.read_csv(rp)
    m = pd.read_csv(cm)
    for df, name in [(r, 'rows'), (m, 'metrics')]:
        assert list(df.columns) == (['dataset','cell','cycle','row_index','ref_cycle','cal_cycle','model','scenario','y_true','y_pred','relative_health'] if name=='rows' else ['dataset','cell','model','scenario','n','mae','rmse','bias']), (stage, name, list(df.columns))
    assert r[['dataset','cell','model','scenario','row_index']].duplicated().sum() == 0
    assert r['cycle'].gt(r['cal_cycle']).all()
    assert r['cal_cycle'].gt(r['ref_cycle']).all()
    r['relative_health'] = r['relative_health'].astype(bool)
    rows.append(r.assign(stage=stage))
    mets.append(m.assign(stage=stage))
    RECEIPTS[stage] = {'rows': int(len(r)), 'cells': sorted(r['cell'].unique().tolist()),
                       'models': sorted(r['model'].unique().tolist()),
                       'scenarios': sorted(r['scenario'].unique().tolist())}
    return r.assign(stage=stage), m.assign(stage=stage)

rows_all, mets_all = [], []
for s in STAGES:
    r, m = load_stage(s)
    rows_all.append(r)
    mets_all.append(m)
rows = pd.concat(rows_all, ignore_index=True)
mets = pd.concat(mets_all, ignore_index=True)

assert set(rows['model'].unique()) == set(KINDS)
assert set(rows['scenario'].unique()) == set(SCENARIOS)
for stage, g in rows.groupby('stage'):
    gc = g.groupby('cell')['model'].nunique()
    assert gc.eq(10).all(), (stage, gc[gc.ne(10)].to_dict())
    for cell, cg in g.groupby('cell'):
        ms = set(cg['model'].unique())
        assert ms == set(KINDS), (stage, cell, sorted(ms))
        for model, mg in cg.groupby('model'):
            scs = set(mg['scenario'].unique())
            if stage == 'loco':
                assert scs == {'clean'}, (stage, cell, model, scs)
            else:
                assert scs == set(SCENARIOS), (stage, cell, model, scs)

rows = rows.copy()
rows['err'] = rows['y_pred'] - rows['y_true']
rows['ae'] = rows['err'].abs()
rows['se'] = rows['err'] ** 2
recomp = rows.groupby(['dataset','cell','model','scenario'], as_index=False).agg(
    n=('ae','size'), mae=('ae','mean'), rmse=('se', lambda x: float(np.sqrt(np.mean(x)))), bias=('err','mean'))
chk = recomp.merge(mets[['dataset','cell','model','scenario','n','mae','rmse','bias']],
                   on=['dataset','cell','model','scenario'], suffixes=('_r','_m'), how='outer')
assert len(chk) == len(recomp) == len(mets)
for a, b in [('n','n'), ('mae','mae'), ('rmse','rmse'), ('bias','bias')]:
    x = chk[f'{a}_r'].to_numpy(dtype=float)
    y = chk[f'{b}_m'].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    assert ok.all(), (a, 'non-finite')
    assert np.allclose(x, y, atol=2e-10, rtol=0), (a, np.nanmax(np.abs(x-y)))

cell_metrics = mets[['dataset','cell','model','scenario','n','mae','rmse','bias']].copy()
cell_metrics['mae_pp'] = cell_metrics['mae'] * 100.0

clean = cell_metrics[cell_metrics['scenario']=='clean'].copy()
clean_cell = clean.groupby(['dataset','cell','model'], as_index=False)['mae_pp'].mean()
piv = clean_cell.pivot_table(index='model', columns='dataset', values='mae_pp', aggfunc='mean')
piv = piv.reindex(KINDS)
assert piv.shape[0] == 10 and piv.shape[1] == 6
assert not piv.isna().any().any()
piv.to_csv(OUT / 'clean_mae_pp.csv', float_format='%.12g')

stress_cell = cell_metrics[cell_metrics['scenario'].isin(STRESS)].groupby(
    ['dataset','model','scenario'], as_index=False)['mae_pp'].mean()
stress_cell = stress_cell.sort_values(['dataset','model','scenario']).reset_index(drop=True)
stress_cell.to_csv(OUT / 'stress_mae_pp.csv', index=False, float_format='%.12g')

clean_rows = rows[rows['scenario']=='clean'].copy()
clean_rows['stratum'] = np.where(clean_rows['y_true'] <= 0.85, 'low_soh', 'above_0.85')
cr = clean_rows.copy()
cr['mae_pp'] = cr['ae'] * 100.0
step = cr.groupby(['dataset','cell','model','stratum'], as_index=False).agg(mae_pp=('mae_pp','mean'), nrows=('ae','size'))
low = step.groupby(['dataset','model','stratum'], as_index=False).agg(
    mae_pp=('mae_pp','mean'), ncells=('cell','nunique'), nrows=('nrows','sum'))
low = low.sort_values(['dataset','model','stratum']).reset_index(drop=True)
low.to_csv(OUT / 'low_soh_mae_pp.csv', index=False, float_format='%.12g')

cell_names = sorted(cell_metrics.loc[cell_metrics.dataset=='XJTU_B5','cell'].unique().tolist())
assert len(cell_names) == 8, len(cell_names)
cell_metrics = cell_metrics.set_index(['dataset','cell','model','scenario'])

def get_cell_mae(dataset, cell, model, scenario='clean'):
    return float(cell_metrics.loc[(dataset, cell, model, scenario), 'mae_pp'])

rng = np.random.default_rng(20261107)
comparators = [k for k in KINDS if k != PRIMARY]
assert len(comparators) == 9
paired_cells_records = []
paired_summary = {}
B = 20000

for comp in comparators:
    for scen in SCENARIOS:
        diffs = []
        pv = []
        cv = []
        cell_tags = []
        for ds in ['XJTU_B5']:
            for c in cell_names:
                try:
                    pm = get_cell_mae(ds, c, PRIMARY, scen)
                    cm = get_cell_mae(ds, c, comp, scen)
                except KeyError as e:
                    raise KeyError(f'missing metric for comp={comp} scen={scen} ds={ds} cell={c}: {e}')
                diffs.append(pm - cm)
                pv.append(pm)
                cv.append(cm)
                cell_tags.append(f'{ds}::{c}')
        diffs = np.asarray(diffs, dtype=float)
        pv = np.asarray(pv, dtype=float)
        cv = np.asarray(cv, dtype=float)
        assert diffs.size == 8 and np.isfinite(diffs).all()
        for tag, d in zip(cell_tags, diffs):
            paired_cells_records.append({'comparison': comp, 'scenario': scen, 'cell': tag, 'diff_pp': d})
        boot = np.empty(B, dtype=float)
        n = diffs.size
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            boot[b] = diffs[idx].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pmean = float(pv.mean())
        cmean = float(cv.mean())
        rel = 1.0 - pmean / cmean if cmean != 0 else float('nan')
        paired_summary[f'{comp}|{scen}'] = {
            'mean_diff_pp': float(diffs.mean()),
            'ci95_pp': [float(lo), float(hi)],
            'wins_primary': int((diffs < 0).sum()),
            'n_cells': int(n),
            'relative_reduction': float(rel),
        }

paired_df = pd.DataFrame(paired_cells_records)
paired_df.to_csv(OUT / 'paired_cells.csv', index=False, float_format='%.12g')
with open(OUT / 'paired_summary.json', 'w') as f:
    json.dump(paired_summary, f, indent=2, sort_keys=True)

conf_path = P / 'stage2' / 'confirmation' / 'confirmation_complete.json'
with open(conf_path) as f:
    conf = json.load(f)
passed = bool(conf.get('passed', conf.get('gate', {}).get('passed', False)))

mean_clean = clean_cell.loc[clean_cell.dataset=='XJTU_B5'].groupby('model')['mae_pp'].mean()
gate = {
    'primary_clean_mean_pp': float(mean_clean[PRIMARY]),
    'isomap_clean_mean_pp': float(mean_clean['isotropic_map']),
    'target_secant_clean_mean_pp': float(mean_clean['target_secant']),
    'best_segment_clean_mean_pp': float(mean_clean['best_segment']),
    'threshold_pp': 0.9,
    'check_isomap': bool(mean_clean[PRIMARY] <= 0.9 * mean_clean['isotropic_map']),
    'check_target_secant': bool(mean_clean[PRIMARY] <= 0.9 * mean_clean['target_secant']),
    'check_best_segment': bool(mean_clean[PRIMARY] <= mean_clean['best_segment']),
    'confirmation_passed': passed,
}
gate['all_passed'] = bool(gate['check_isomap'] and gate['check_target_secant'] and gate['check_best_segment'] and passed)
assert bool(gate['check_isomap'] and gate['check_target_secant'] and gate['check_best_segment']) == passed

# figure 1: clean heatmap 10x6 annotated
fig, ax = plt.subplots(figsize=(7.2, 7.2))
vals = piv.to_numpy(dtype=float)
cmap = plt.get_cmap('viridis')
norm = Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
im = ax.imshow(vals, aspect='auto', cmap=cmap, norm=norm)
ax.set_xticks(np.arange(piv.shape[1]))
ax.set_xticklabels([str(c) for c in piv.columns], rotation=0)
ax.set_yticks(np.arange(len(KINDS)))
ax.set_yticklabels(KINDS)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        r, g, b, _ = cmap(norm(vals[i, j]))
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        color = 'white' if lum < 0.5 else 'black'
        ax.text(j, i, f'{vals[i, j]:.2f}', ha='center', va='center', color=color, fontsize=7)
ax.set_title('Clean MAE (pp) by model and dataset')
ax.set_xlabel('Dataset')
ax.set_ylabel('Model')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='MAE (pp)')
fig.tight_layout()
fig.savefig(FIG / 'clean_heatmap.png', dpi=300)
plt.close(fig)

# figure 2: paired errors for primary vs 4 selected comparators across 5 scenarios
sel = ['primary','isotropic_map', 'source_logridge', 'scalar_map', 'target_secant']
sel_labels = sel
fig, axes = plt.subplots(1, 5, figsize=(7.2, 4.2), sharey=True)
xs = np.arange(1, 9)
markers = ['o', 's', '^', 'D', 'v']
for ax, scen in zip(axes, SCENARIOS):
    for mi, comp in enumerate(sel):
        key = f'{comp}|{scen}'
        cell_vals = []
        for c in cell_names:
            ds = sorted(cell_metrics.index.get_level_values('dataset').unique())[0]
            vals_c = []
            for d in ['XJTU_B5']:
                try:
                    vals_c.append(get_cell_mae(d, c, comp, scen))
                except KeyError:
                    pass
            if vals_c:
                cell_vals.append(float(np.mean(vals_c)))
        if len(cell_vals) == 8:
            ax.plot(xs, cell_vals, marker=markers[mi % len(markers)], linewidth=1.0, markersize=3.5, label=sel_labels[mi])
    ax.set_title(scen)
    ax.set_xticks(xs)
    if scen == 'clean':
        ax.set_ylabel('Cell mean MAE (pp)')
    ax.grid(alpha=0.25, linewidth=0.5)
axes[0].legend(fontsize=6, frameon=False)
fig.suptitle('B5 paired cell errors: comparator model MAE (pp)')
fig.tight_layout()
fig.savefig(FIG / 'b5_paired.png', dpi=300)
plt.close(fig)

counts = {
    'cohort_ncells': rows.groupby('dataset').cell.nunique().to_dict(),
    'unique_profiles': int(rows[['dataset','cell','cycle','row_index']].drop_duplicates().shape[0]),
    'cal_cycles_per_cell': {f'{ds}::{c}': float(g.cal_cycle.iloc[0]) for (ds,c),g in rows.loc[rows.model==PRIMARY].groupby(['dataset','cell'])},
    'methods': len(KINDS),
    'scenarios': SCENARIOS,
}

summary = {
    'units': 'relative_capacity_pp',
    'note': 'B1 nested LOCO vs B2/B3/B4/B6 frozen development vs B5 once confirmation; primary must beat comparator mean MAE by gate; paired bootstrap over cells with seed 20261107.',
    'table_clean_records': piv.reset_index().to_dict('records'),
    'paired_summary_keys': len(paired_summary),
    'gate': gate,
    'counts': counts,
    'clean_b5_means_pp': {k: float(v) for k, v in mean_clean.to_dict().items()},
    'receipts': RECEIPTS,
}
with open(OUT / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print('COMPLETED analysis_stage2.py')
print(f'cells={len(cell_names)} records={len(rows)} metrics={len(mets)}')
print(f'clean B5 means (pp): ' + ', '.join(f'{k}={v:.4f}' for k, v in mean_clean.to_dict().items()))
print(f'gate all_passed={gate["all_passed"]} confirmation_passed={passed}')
print(f'outputs written to {OUT}')

