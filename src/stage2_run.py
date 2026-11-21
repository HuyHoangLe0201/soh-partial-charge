import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import sys, json, math, hashlib, time, traceback
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Any, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.model_selection import GroupKFold
import reference_models as ref

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'paper_research_expansion_20261107' / 'data'
NEW = ROOT / 'data'
STAGE2 = ROOT / 'stage2'
LAMBDAS = [1e-6, 1e-4, 0.01, 0.1]
TG = [0.01, 0.1, 1.0, 10.0]
STRIDES = [1, 6, 11]
SCEN = ['clean','vp5','vm5','vp10','vm10']
KINDS = ['primary','isotropic_map','target_secant','endpoint_secant','source_logridge','scalar_map','carry_last','cycle_trend','covariance_exact','best_segment']

def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for ch in iter(lambda: f.read(1<<20), b''):
            h.update(ch)
    return h.hexdigest()

def npz_path(ds: str) -> Path:
    if ds == 'XJTU_B1': return OLD/'XJTU_B1.npz'
    if ds in ('XJTU_B2','XJTU_B3'): return OLD/f'{ds}.npz'
    if ds in ('XJTU_B4','XJTU_B5','XJTU_B6'): return NEW/f'{ds}.npz'
    raise ValueError(ds)

def load(ds: str) -> List[Any]:
    p = npz_path(ds)
    if not p.exists(): raise FileNotFoundError(p)
    return ref.load_cells(str(p), ds)

def rel_parts(c): 
    X = np.log(np.asarray(c.D['clean'],float)/np.asarray(c.Dref['clean'],float)[None,:])
    z = np.log(np.asarray(c.y,float)/float(c.ref_y))
    return X, z

def fit_stats(TRAIN: List[Any], lam: float) -> Dict[str, Any]:
    lam = float(lam)
    Hs, hs, bcs, resids = [], [], [], []
    for c in TRAIN:
        X, z = rel_parts(c)
        if X.shape[0] < 1: raise ValueError(f'{c.name}: no future rows')
        Hc = (X.T @ X)/X.shape[0]
        hc = (X.T @ z)/X.shape[0]
        bc = np.linalg.solve(Hc + lam*np.eye(4), hc)
        if not np.all(np.isfinite(bc)): raise FloatingPointError(f'{c.name}: nonfinite beta_c')
        Hs.append(Hc); hs.append(hc); bcs.append(bc); resids.append(float(np.mean((z - X@bc)**2)))
    n = len(TRAIN)
    H = np.mean(np.stack(Hs,0),0); h = np.mean(np.stack(hs,0),0)
    beta0 = np.linalg.solve(H + lam*np.eye(4), h)
    Bc = np.stack(bcs,0); mu = Bc.mean(0); Dm = Bc - mu[None,:]
    S = (Dm.T @ Dm)/Bc.shape[0]
    tr = float(np.trace(S))
    Sigma = S + (0.1*tr/4.0 + 1e-6)*np.eye(4)
    sigma2 = max(float(np.mean(resids)), 1e-6)
    if not (np.all(np.isfinite(beta0)) and np.all(np.isfinite(Sigma)) and math.isfinite(sigma2)):
        raise FloatingPointError('fit_stats nonfinite')
    return {'lambda':lam,'H':H.tolist(),'h':h.tolist(),'beta0':beta0.tolist(),'S':S.tolist(),'Sigma':Sigma.tolist(),'sigma2':sigma2,'beta_c':Bc.tolist(),'cell_names':[c.name for c in TRAIN]}

def _stats_arrays(st):
    return (np.asarray(st['beta0'],float), np.asarray(st['Sigma'],float), float(st['sigma2']), np.asarray(st['H'],float), np.asarray(st['h'],float))

def predict(kind: str, stats: Dict[str,Any], params: Dict[str,Any], cell: Any, tag: str='clean', stride: int=1) -> Dict[str,Any]:
    if stride < 1: raise ValueError('stride')
    Xtag = np.log(np.asarray(cell.D[tag],float)/np.asarray(cell.Dref[tag],float)[None,:])
    nfull = Xtag.shape[0] + 1
    full_idx = np.arange(0, nfull, stride)
    if full_idx.size < 6:
        raise RuntimeError(f'{cell.name}: stride {stride} yields {full_idx.size} full profiles (<6 required)')
    cal_full = int(full_idx[4]); fut_full = full_idx[5:]
    cal_pos = cal_full - 1
    fut_pos = fut_full - 1
    if cal_pos < 0 or np.any(fut_pos < 0): raise RuntimeError('bad stride indexing vs Cell.future')
    xa = Xtag[cal_pos].copy(); za = math.log(float(cell.y[cal_pos])/float(cell.ref_y))
    Xf = Xtag[fut_pos].copy()
    beta0, Sigma, sigma2, H, h = _stats_arrays(stats)
    lam = float(params.get('lambda', stats['lambda']))
    t = float(params.get('t', 1.0))
    segment = int(params.get('segment', 0))
    meta = {'cal_position':int(cal_pos),'cal_cycle':float(cell.cycles[cal_pos]) if hasattr(cell,'cycles') and len(cell.cycles)>cal_pos else float(cell.ref_cycle),'beta':None,'segment':segment,'lambda':lam,'t':t,'stride':int(stride)}
    def carry():
        p = np.full(fut_pos.shape, math.exp(za), float)
        meta['carry'] = True
        return p
    if not (np.all(np.isfinite(xa)) and math.isfinite(za)): raise FloatingPointError('cal nonfinite')
    beta = None
    if kind == 'primary':
        den = t*sigma2 + float(xa@Sigma@xa)
        if den <= 0: raise FloatingPointError('primary denom')
        beta = beta0 + Sigma@xa*(za - float(xa@beta0))/den
        pred = np.exp(Xf@beta)
    elif kind == 'isotropic_map':
        iso = (float(np.trace(Sigma))/4.0)*np.eye(4)
        den = t*sigma2 + float(xa@iso@xa)
        if den <= 0: raise FloatingPointError('iso denom')
        beta = beta0 + iso@xa*(za - float(xa@beta0))/den
        pred = np.exp(Xf@beta)
    elif kind == 'scalar_map':
        f = float(xa@beta0); a = 1.0 + f*(za - f)/(t*sigma2 + f*f)
        beta = a*beta0
        pred = np.exp(Xf@beta)
    elif kind == 'target_secant':
        nrm = float(xa@xa)
        if nrm <= 1e-10: return {'pred':carry(),'future_positions':fut_pos,'metadata':meta}
        beta = xa*za/nrm
        pred = np.exp(Xf@beta)
    elif kind == 'endpoint_secant':
        Dc = np.asarray(cell.D[tag],float)[cal_pos]; Dr = np.asarray(cell.Dref[tag],float)
        ua = float(np.log(np.sum(Dc)/np.sum(Dr)))
        if abs(ua) <= 1e-8: return {'pred':carry(),'future_positions':fut_pos,'metadata':meta}
        pcoef = za/ua
        Du = np.asarray(cell.D[tag],float)[fut_pos]; Dr_sum = float(np.sum(Dr))
        u_t = np.log(np.sum(Du,axis=1)/Dr_sum)
        pred = np.exp(pcoef*u_t)
    elif kind == 'source_logridge':
        b = np.linalg.solve(H + lam*np.eye(4), h)
        beta = b
        pred = np.exp(Xf@beta)
    elif kind == 'carry_last':
        return {'pred':carry(),'future_positions':fut_pos,'metadata':meta}
    elif kind == 'cycle_trend':
        ccal = float(cell.cycles[cal_pos]) if hasattr(cell,'cycles') and len(cell.cycles)>cal_pos else float(cell.ref_cycle)
        cref = float(cell.ref_cycle)
        den = (ccal - cref)
        if abs(den) <= 1e-12:
            return {'pred':carry(),'future_positions':fut_pos,'metadata':meta}
        expza = math.exp(za)
        cfut = np.asarray(cell.cycles,float)[fut_pos]
        rh = 1.0 + (expza - 1.0)*(cfut - cref)/den
        pred = rh
        meta['cal_cycle'] = ccal
        meta['carry'] = False
        if not np.all(np.isfinite(pred)): raise FloatingPointError('cycletrend nonfinite')
        return {'pred':pred,'future_positions':fut_pos,'metadata':meta}
    elif kind == 'covariance_exact':
        den = float(xa@Sigma@xa)
        if den <= 1e-10: return {'pred':carry(),'future_positions':fut_pos,'metadata':meta}
        beta = beta0 + Sigma@xa*(za - float(xa@beta0))/den
        pred = np.exp(Xf@beta)
    elif kind == 'best_segment':
        j = int(segment)
        if j < 0 or j > 3: raise ValueError('segment')
        xaj = float(xa[j])
        if abs(xaj) <= 1e-8: return {'pred':carry(),'future_positions':fut_pos,'metadata':meta}
        bj = za/xaj
        beta = np.zeros(4); beta[j] = bj
        pred = np.exp(Xf@beta)
    else:
        raise ValueError(kind)
    if beta is not None: meta['beta'] = np.asarray(beta,float).tolist()
    if not np.all(np.isfinite(pred)): raise FloatingPointError(f'{kind}: nonfinite pred')
    return {'pred':pred,'future_positions':fut_pos,'metadata':meta}

def _cand_grid(kind: str) -> List[Dict[str,Any]]:
    if kind in ('primary','isotropic_map','scalar_map'):
        return [{'lambda':float(l),'t':float(t)} for l in LAMBDAS for t in TG]
    if kind == 'source_logridge':
        return [{'lambda':float(l)} for l in LAMBDAS]
    if kind == 'covariance_exact':
        return [{'lambda':float(l)} for l in LAMBDAS]
    if kind == 'best_segment':
        return [{'segment':int(j),'lambda':1e-6} for j in range(4)]
    return [{}]

def _tie_key(k):
    return (float(k.get('lambda', -1.0)), float(k.get('t', 0.0)), int(k.get('segment', -1)))

def select(kind: str, TRAIN: List[Any], verbose: bool=True) -> Dict[str,Any]:
    names = sorted(c.name for c in TRAIN)
    cells = sorted(TRAIN, key=lambda c:c.name)
    n = len(cells)
    if n < 3: raise ValueError('select needs >=3')
    gk = GroupKFold(n_splits=3)
    idx = np.arange(n)
    folds = list(gk.split(idx, groups=idx))
    grid = _cand_grid(kind)
    best = None; trace = []
    for cand in grid:
        fold_scores = []
        ok = True
        for tr_i, va_i in folds:
            tr = [cells[i] for i in tr_i]
            lam = float(cand.get('lambda', 1e-6))
            try:
                st = fit_stats(tr, lam)
            except Exception:
                raise
            for vi in va_i:
                vc = cells[vi]
                for s in STRIDES:
                    try:
                        out = predict(kind, st, cand, vc, 'clean', s)
                    except Exception as e:
                        raise RuntimeError(f'bad candidate {kind} {cand} stride {s} cell {vc.name}: {e}')
                    if 'future_positions' not in out: raise RuntimeError('predict missing future_positions')
                    ytrue = np.asarray(vc.y,float)[out['future_positions']]/float(vc.ref_y)
                    mae = float(np.mean(np.abs(ytrue - out['pred'])))
                    if not math.isfinite(mae): raise FloatingPointError('nonfinite MAE')
                    fold_scores.append(mae)
                    trace.append({'kind':kind,'params':cand,'fold_train_names':[c.name for c in tr],'val_cell':vc.name,'stride':int(s),'mae':mae})
        if not ok: continue
        score = float(np.mean(fold_scores))
        key = (score,) + _tie_key(cand)
        if best is None or key < best[0]:
            best = (key, cand, score)
        if verbose: print(f'[{kind}] {cand} score={score:.6g}', flush=True)
    if best is None: raise RuntimeError(f'no valid candidate for {kind}')
    return {'kind':kind,'chosen':best[1],'score':best[2],'trace':trace}

def _score_save(rows: List[Dict[str,Any]], dirp: Path, prefix: str) -> Dict[str,Any]:
    dirp.mkdir(parents=True, exist_ok=True)
    (dirp/f'{prefix}_row_predictions.csv').write_text(
        'dataset,cell,cycle,row_index,ref_cycle,cal_cycle,model,scenario,y_true,y_pred,relative_health\n' +
        '\n'.join(f"{r['dataset']},{r['cell']},{r['cycle']},{r['row_index']},{r['ref_cycle']},{r['cal_cycle']},{r['model']},{r['scenario']},{r['y_true']},{r['y_pred']},True" for r in rows), encoding='utf-8')
    by = {}
    for r in rows:
        by.setdefault((r['dataset'],r['cell'],r['model'],r['scenario']),[]).append(r)
    cell_metrics = []
    for k, rs in by.items():
        ds,cell,model,sc = k
        y = np.array([x['y_true'] for x in rs],float); p = np.array([x['y_pred'] for x in rs],float)
        cell_metrics.append({'dataset':ds,'cell':cell,'model':model,'scenario':sc,'n':len(rs),'mae':float(np.mean(np.abs(p-y))),'rmse':float(np.sqrt(np.mean((p-y)**2))),'bias':float(np.mean(p-y))})
    (dirp/f'{prefix}_cell_metrics.csv').write_text(
        'dataset,cell,model,scenario,n,mae,rmse,bias\n' +
        '\n'.join(f"{m['dataset']},{m['cell']},{m['model']},{m['scenario']},{m['n']},{m['mae']},{m['rmse']},{m['bias']}" for m in cell_metrics), encoding='utf-8')
    summ = {'cell_metrics':cell_metrics}
    (dirp/f'{prefix}_summary.json').write_text(json.dumps(summ, indent=2), encoding='utf-8')
    return summ

def _params_for(kind: str, chosen: Dict[str,Any], sel: Dict[str,Any]) -> Dict[str,Any]:
    p = dict(chosen)
    if 'lambda' not in p: p['lambda'] = 1e-6
    return p

def do_source():
    src = STAGE2/'source'
    src.mkdir(parents=True, exist_ok=True)
    conf_start = STAGE2/'confirmation_started'
    if conf_start.exists(): raise RuntimeError('confirmation_started exists: refuse section write')
    freeze = src/'config_freeze.json'
    if freeze.exists(): raise RuntimeError('freeze exists: refuse overwrite')
    b1p = npz_path('XJTU_B1')
    b1h = _sha(b1p)
    b1 = load('XJTU_B1')
    if len(b1) != 8: raise RuntimeError('B1 requires exactly 8 cells')
    models = {}
    for kind in KINDS:
        print(f'select {kind}', flush=True)
        s = select(kind, b1)
        (src/f'candidate_trace_{kind}.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
        chosen = s['chosen']
        lam = float(chosen.get('lambda',1e-6))
        st = fit_stats(b1, lam)
        models[kind] = {'kind':kind,'chosen':chosen,'lambda':lam,'stats':st,'params':_params_for(kind, chosen, s)}
        (src/f'candidate_rows_{kind}.csv').write_text(
            'kind,val_cell,stride,lambda,t,segment,mae\n' +
            '\n'.join(f"{r['kind']},{r['val_cell']},{r['stride']},{r['params'].get('lambda','')},{r['params'].get('t','')},{r['params'].get('segment','')},{r['mae']}" for r in s['trace']), encoding='utf-8')
    code_files = ['stage2_run.py','stage2_spec.txt','reference_models.py']
    code_hash = {}
    for f in code_files:
        p = ROOT/f
        if p.exists(): code_hash[f] = _sha(p)
    cfg = {'written_epoch':time.time(),'criterion':{'reduction_vs_isotropic_map':0.1,'reduction_vs_target_secant':0.1,'not_worse_than_source_selected_best_segment':True,'descriptive_not_statistical':True},'b1_npz':'XJTU_B1.npz','b1_sha256':b1h,'code_sha256':code_hash,'models':models}
    (src/'config_freeze.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print('FREEZE COMPLETE', flush=True)

def _verify_freeze() -> Dict[str,Any]:
    cfgp = STAGE2/'source'/'config_freeze.json'
    if not cfgp.exists(): raise RuntimeError('no freeze')
    cfg = json.loads(cfgp.read_text())
    if _sha(npz_path('XJTU_B1')) != cfg['b1_sha256']: raise RuntimeError('B1 hash mismatch')
    for f,h in cfg.get('code_sha256',{}).items():
        p = ROOT/f
        if not p.exists(): raise RuntimeError('missing code file')
        if _sha(p) != h: raise RuntimeError(f'code hash mismatch {f}')
    return cfg

def _predict_dataset(cfg: Dict[str,Any], ds: str, outdir: Path, prefix: str):
    cells = load(ds)
    rows = []
    for c in cells:
        nfull = np.asarray(c.y).size + 1
        for model in KINDS:
            m = cfg['models'][model]
            st = m['stats']; params = m['params']
            for sc in SCEN:
                if False:  # rejected proposal's target segment sweep is disabled
                    for j in range(4):
                        pj = dict(params); pj['segment'] = j
                        out = predict(model, st, pj, c, sc, 1)
                        fp = out['future_positions']
                        for k,pos in enumerate(fp):
                            rows.append({'dataset':ds,'cell':c.name,'cycle':float(c.cycles[pos]),'row_index':int(c.row_index[pos]) if hasattr(c,'row_index') else int(pos),'ref_cycle':float(c.ref_cycle),'cal_cycle':float(out['metadata']['cal_cycle']),'model':model+f'_j{j}','scenario':sc,'y_true':float(c.y[pos])/float(c.ref_y),'y_pred':float(out['pred'][k])})
                else:
                    out = predict(model, st, params, c, sc, 1)
                    fp = out['future_positions']
                    for k,pos in enumerate(fp):
                        rows.append({'dataset':ds,'cell':c.name,'cycle':float(c.cycles[pos]),'row_index':int(c.row_index[pos]) if hasattr(c,'row_index') else int(pos),'ref_cycle':float(c.ref_cycle),'cal_cycle':float(out['metadata']['cal_cycle']),'model':model,'scenario':sc,'y_true':float(c.y[pos])/float(c.ref_y),'y_pred':float(out['pred'][k])})
    return _score_save(rows, outdir, prefix)

def do_development():
    cfg = _verify_freeze()
    dev = STAGE2/'development'; dev.mkdir(parents=True, exist_ok=True)
    if (dev/'development_complete.json').exists(): raise RuntimeError('development exists refuse overwrite')
    all_rows = []
    for ds in ['XJTU_B2','XJTU_B3','XJTU_B4','XJTU_B6']:
        print(f'dev {ds}', flush=True)
        cells = load(ds)
        for c in cells:
            for model in KINDS:
                m = cfg['models'][model]; st=m['stats']; params=m['params']
                for sc in SCEN:
                    if False:  # rejected proposal's target segment sweep is disabled
                        for j in range(4):
                            pj=dict(params); pj['segment']=j
                            out = predict(model, st, pj, c, sc, 1)
                            for k,pos in enumerate(out['future_positions']):
                                all_rows.append({'dataset':ds,'cell':c.name,'cycle':float(c.cycles[pos]),'row_index':int(c.row_index[pos]),'ref_cycle':float(c.ref_cycle),'cal_cycle':float(out['metadata']['cal_cycle']),'model':model+f'_j{j}','scenario':sc,'y_true':float(c.y[pos])/float(c.ref_y),'y_pred':float(out['pred'][k])})
                    else:
                        out = predict(model, st, params, c, sc, 1)
                        for k,pos in enumerate(out['future_positions']):
                            all_rows.append({'dataset':ds,'cell':c.name,'cycle':float(c.cycles[pos]),'row_index':int(c.row_index[pos]),'ref_cycle':float(c.ref_cycle),'cal_cycle':float(out['metadata']['cal_cycle']),'model':model,'scenario':sc,'y_true':float(c.y[pos])/float(c.ref_y),'y_pred':float(out['pred'][k])})
    summ = _score_save(all_rows, dev, 'development')
    (dev/'development_complete.json').write_text(json.dumps({'n_rows':len(all_rows)}, indent=2), encoding='utf-8')

def do_confirmation():
    cfg = _verify_freeze()
    conf = STAGE2/'confirmation'; conf.mkdir(parents=True, exist_ok=True)
    started = STAGE2/'confirmation_started'
    if (conf/'confirmation_complete.json').exists(): raise RuntimeError('confirmation already complete')
    if started.exists(): raise RuntimeError('confirmation_started already exists')
    started.write_text(json.dumps({'time':time.time(),'stage':'confirmation'}), encoding='utf-8')
    cells = load('XJTU_B5')
    if len(cells) != 8: raise RuntimeError(f'B5 cells {len(cells)} != 8')
    for c in cells:
        if np.asarray(c.y).size < 5: raise RuntimeError(f'{c.name}: future rows < 5')
    rows = []
    for c in cells:
        for model in KINDS:
            m = cfg['models'][model]; st=m['stats']; params=m['params']
            for sc in SCEN:
                if False:  # rejected proposal's target segment sweep is disabled
                    for j in range(4):
                        pj=dict(params); pj['segment']=j
                        out = predict(model, st, pj, c, sc, 1)
                        for k,pos in enumerate(out['future_positions']):
                            rows.append({'dataset':'XJTU_B5','cell':c.name,'cycle':float(c.cycles[pos]),'row_index':int(c.row_index[pos]),'ref_cycle':float(c.ref_cycle),'cal_cycle':float(out['metadata']['cal_cycle']),'model':model+f'_j{j}','scenario':sc,'y_true':float(c.y[pos])/float(c.ref_y),'y_pred':float(out['pred'][k])})
                else:
                    out = predict(model, st, params, c, sc, 1)
                    for k,pos in enumerate(out['future_positions']):
                        rows.append({'dataset':'XJTU_B5','cell':c.name,'cycle':float(c.cycles[pos]),'row_index':int(c.row_index[pos]),'ref_cycle':float(c.ref_cycle),'cal_cycle':float(out['metadata']['cal_cycle']),'model':model,'scenario':sc,'y_true':float(c.y[pos])/float(c.ref_y),'y_pred':float(out['pred'][k])})
    summ = _score_save(rows, conf, 'confirmation')
    by = {(m['cell'],m['model'],m['scenario']):m for m in summ['cell_metrics']}
    def mae(model):
        vals = [by.get((c.name, model, 'clean'),{}).get('mae') for c in cells]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float('nan')
    m_pri = mae('primary'); m_iso = mae('isotropic_map'); m_sec = mae('target_secant'); m_best = mae('best_segment')
    pass10 = (m_pri <= 0.9*m_iso) and (m_pri <= 0.9*m_sec)
    passbest = (m_pri <= m_best) if (math.isfinite(m_best) and math.isfinite(m_pri)) else False
    passed = bool(pass10 and passbest)
    comp = {'passed':passed,'primary_clean_mae':m_pri,'isomap_clean_mae':m_iso,'target_secant_clean_mae':m_sec,'best_segment_clean_mae':m_best,'reduction_vs_isomap':(1-m_pri/m_iso) if m_iso else None,'reduction_vs_target_secant':(1-m_pri/m_sec) if m_sec else None,'n_rows':len(rows)}
    (conf/'confirmation_complete.json').write_text(json.dumps(comp, indent=2), encoding='utf-8')
    if not passed: print('CONFIRMATION GATE FAILED (report only)', flush=True)

def do_loco():
    _verify_freeze()
    b1 = load('XJTU_B1')
    out = STAGE2/'loco'; out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for i,c in enumerate(b1):
        tr = [d for j,d in enumerate(b1) if j!=i]
        for kind in KINDS:
            s = select(kind, tr, verbose=False)
            lam = float(s['chosen'].get('lambda',1e-6))
            st = fit_stats(tr, lam)
            params = _params_for(kind, s['chosen'], s)
            try:
                o = predict(kind, st, params, c, 'clean', 1)
            except Exception as e:
                raise RuntimeError(f'loco {c.name} {kind}: {e}')
            for k,pos in enumerate(o['future_positions']):
                rows.append({'dataset':'XJTU_B1','cell':c.name,'cycle':float(c.cycles[pos]),'row_index':int(c.row_index[pos]),'ref_cycle':float(c.ref_cycle),'cal_cycle':float(o['metadata']['cal_cycle']),'model':kind,'scenario':'clean','y_true':float(c.y[pos])/float(c.ref_y),'y_pred':float(o['pred'][k])})
    _score_save(rows, out, 'loco')

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['source','development','confirmation','loco'])
    a = ap.parse_args()
    try:
        if a.stage == 'source': do_source()
        elif a.stage == 'development': do_development()
        elif a.stage == 'confirmation': do_confirmation()
        elif a.stage == 'loco': do_loco()
        print('OK', a.stage, flush=True)
        return 0
    except Exception as e:
        print('ERROR', a.stage, repr(e), flush=True)
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        raise SystemExit(main())
