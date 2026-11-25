"""CPU-only reproduction of the E16 paired analysis; never calls a model/API."""
import argparse, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='outputs/recomputed_e16.json');a=ap.parse_args()
    base=ROOT/'results/e16'
    rows=[json.loads(s) for s in (base/'scored_records.jsonl').read_text().splitlines()]
    saved=json.loads((base/'summary.json').read_text());records=json.loads((base/'records.json').read_text())
    ids=sorted(r['id'] for r in records);domains={r['id']:r['domain'] for r in records}
    assert len(rows)==144 and len(ids)==len(set(ids))==36
    cells={(b,m):{r['id']:r for r in rows if r['budget']==b and r['arm']==m} for b in [8192,40960] for m in ['base','best']}
    for (budget,arm),cell in cells.items():
        assert set(cell)==set(ids)
        for r in cell.values():
            assert r['N']==4 and len(r['keys'])==4 and r['L']>0
            assert np.isclose(r['VR'],sum(k is not None for k in r['keys'])/4)
            assert np.isclose(r['RR'],len({k for k in r['keys'] if k is not None})/r['L'])
            assert np.isclose(r['tokens_mean'],np.mean(r['lengths']))
            assert np.isclose(r['truncation_rate'],1-np.mean(r['ended']))
        for k in ['VR','NR','RR','tokens_mean','truncation_rate']:
            value=np.mean([r[k] for r in cell.values()])
            assert np.isclose(value,saved['cells'][f'{budget}_{arm}']['domain_macro'][k]),(budget,arm,k)
    rng=np.random.default_rng(20261121)
    blocks=[np.array([j for j,i in enumerate(ids) if domains[i]==d]) for d in ['causal','3d','boolean']]
    indices=np.concatenate([rng.choice(b,size=(20000,12),replace=True) for b in blocks],axis=1)
    signs=np.random.default_rng(20261122).integers(0,2,size=(100000,36),dtype=np.int8)*2-1
    comparisons={}
    for b in [8192,40960]:
        comp={}
        for k in ['RR','VR','NR','tokens_mean','truncation_rate']:
            d=np.array([cells[b,'best'][i][k]-cells[b,'base'][i][k] for i in ids])
            v={'delta':float(d.mean()),'CI95':np.quantile(d[indices].mean(axis=1),[.025,.975]).tolist()}
            original=saved['best_minus_base'][str(b)][k]
            assert np.allclose(v['CI95'],original['paired_domain_stratified_bootstrap_CI95'])
            if k=='RR':v['p']=float((1+np.sum(np.abs(signs@d)>=abs(d.sum())-1e-12))/100001)
            comp[k]=v
        comparisons[str(b)]=comp
    ordered=sorted(comparisons,key=lambda b:comparisons[b]['RR']['p']);previous=0
    for rank,b in enumerate(ordered):
        previous=max(previous,min(1,(2-rank)*comparisons[b]['RR']['p']))
        comparisons[b]['RR']['Holm_p']=previous
        assert np.isclose(previous,saved['best_minus_base'][b]['RR']['Holm_adjusted_p_over_two_budgets'])
    result={'verified':True,'rows':144,'questions':36,'comparisons':comparisons}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print('Verified 144 score records, cell means, paired intervals, and Holm-adjusted RR comparisons.')
    for b in comparisons:
        rr=comparisons[b]['RR'];t=comparisons[b]['tokens_mean']
        print(f"{b}: RR delta {rr['delta']*100:+.2f} pp, Holm p={rr['Holm_p']:.3f}; token delta {t['delta']:.1f}, CI={t['CI95']}")
if __name__=='__main__':main()
