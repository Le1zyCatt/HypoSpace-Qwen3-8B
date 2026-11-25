"""Single-GPU, batch-one E16 protocol reimplementation; no remote orchestration."""
import argparse,hashlib,json,time
from pathlib import Path
from reward import candidate_key,parse_key

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--arm',choices=['base','best'],required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=36);a=p.parse_args()
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,set_seed
    root=Path(__file__).resolve().parent.parent
    records=json.loads((root/'results/e16/records.json').read_text())[:a.limit]
    cfg=AutoConfig.from_pretrained(a.model,local_files_only=True)
    # As in E16: extend capacity without changing original default RoPE. This is not YaRN.
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    before,_=Qwen3RotaryEmbedding.compute_default_rope_parameters(cfg,device='cpu')
    assert cfg.rope_parameters['rope_type']=='default'
    cfg.max_position_embeddings=41745
    after,_=Qwen3RotaryEmbedding.compute_default_rope_parameters(cfg,device='cpu');assert torch.equal(before,after)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side='left');tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(a.model,config=cfg,local_files_only=True,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa');m.eval()
    outdir=Path(a.output);outdir.mkdir(parents=True,exist_ok=True)
    for r in records:
        destination=outdir/(r['id']+'.json')
        if destination.exists():continue
        prompt=tok.apply_chat_template(r['prompt'],tokenize=False,add_generation_prompt=True,enable_thinking=True);parts=[]
        for i in range(4):
            seed=(20261113+int(hashlib.sha256(f"confirm:{r['id']}:{i}".encode()).hexdigest()[:8],16))%2**32
            set_seed(seed);inputs=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda');t=time.time()
            with torch.inference_mode():out=m.generate(**inputs,do_sample=True,temperature=.6,top_p=.95,top_k=20,min_p=0.,max_new_tokens=40960,cache_implementation='dynamic',eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True,logits_to_keep=1)
            ids=out[0,inputs['input_ids'].shape[1]:].tolist();end=next((j+1 for j,v in enumerate(ids) if v==tok.eos_token_id),len(ids));ids=ids[:end]
            parts.append({'seed':seed,'ids':ids,'seconds':time.time()-t,'prompt_tokens':inputs['input_ids'].shape[1]})
        result={'id':r['id'],'domain':r['domain'],'arm':a.arm,'L':r['L'],'N':4,'parts':parts,'cells':{}}
        for budget in [8192,40960]:
            ids=[x['ids'][:budget] for x in parts];raw=[tok.decode(x,skip_special_tokens=True) for x in ids]
            final=[x.rsplit('</think>',1)[1].strip() if '</think>' in x else '' for x in raw]
            keys=[candidate_key(x,r) for x in final];parsed=[parse_key(x,r) for x in final]
            result['cells'][budget]={'final':final,'keys':keys,'VR':sum(k is not None for k in keys)/4,'NR':len({k for k in parsed if k is not None})/4,'RR':len({k for k in keys if k is not None})/r['L'],'tokens_mean':sum(map(len,ids))/4,'truncation_rate':sum(x[-1]!=tok.eos_token_id for x in ids)/4}
        destination.write_text(json.dumps(result,ensure_ascii=False))
        print(r['id'],{k:round(v['RR'],4) for k,v in result['cells'].items()},flush=True)
if __name__=='__main__':main()
