"""Re-score archived final strings, and optionally decode exact paired prefixes on CPU."""
import argparse,gzip,json
from pathlib import Path
from reward import candidate_key,parse_key
ROOT=Path(__file__).resolve().parent.parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--tokenizer',help='Local original Qwen3-8B tokenizer; also validates prefix decoding');a=p.parse_args()
    b=ROOT/'results/e16';records={r['id']:r for r in json.loads((b/'records.json').read_text())}
    rows=[json.loads(s) for s in (b/'scored_records.jsonl').read_text().splitlines()]
    for r in rows:
        q=records[r['id']];keys=[candidate_key(x,q) for x in r['final']];parsed=[parse_key(x,q) for x in r['final']]
        assert keys==r['keys'];assert len({x for x in parsed if x is not None})==r['unique_parsed']
        assert len({x for x in keys if x is not None})==r['unique_valid']
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
        with gzip.open(b/'trajectories.jsonl.gz','rt') as f:parts=[json.loads(s) for s in f]
        parts={(x['arm'],x['id'],x['sample']):x for x in parts}
        for r in rows:
            for i in range(4):
                part=parts[r['arm'],r['id'],i];ids=part['ids'][:r['budget']]
                assert len(ids)==r['lengths'][i] and part['seed']==r['seeds'][i]
                raw=tok.decode(ids,skip_special_tokens=True)
                final=raw.rsplit('</think>',1)[1].strip() if '</think>' in raw else ''
                assert final==r['final'][i]
    print('Verified canonical parsing, strict GT membership, deduplication'+(' and all 576 paired prefixes.' if a.tokenizer else '. Prefix decoding not requested.'))
if __name__=='__main__':main()
