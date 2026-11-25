"""Portable E16 retraining entry; consumes the original ordered tokenized data.
GPU reproduction entry, not a claim of an additional training run.
"""
import argparse,gzip,json,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    os.environ['MODEL_PATH']=a.model;os.environ['TOKENIZERS_PARALLELISM']='false'
    import training_core as core
    from transformers import AutoTokenizer,TrainingArguments,DataCollatorForSeq2Seq,set_seed
    from datasets import Dataset
    root=Path(__file__).resolve().parent.parent
    with gzip.open(root/'data/sft_data_diverse.json.gz','rt') as f:rows=json.load(f)
    assert len(rows)==576 and len({r['id'] for r in rows})==48
    assert max(len(r['input_ids']) for r in rows)==5343
    set_seed(33550336)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side='right');tok.pad_token=tok.eos_token
    model=core.make_model()
    data=Dataset.from_list([{k:r[k] for k in ['input_ids','attention_mask','labels']} for r in rows])
    args=TrainingArguments(output_dir=a.output,num_train_epochs=1,per_device_train_batch_size=1,gradient_accumulation_steps=8,learning_rate=2e-5,lr_scheduler_type='cosine',warmup_steps=4,weight_decay=0.,max_grad_norm=1.,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},optim='adamw_torch',seed=33550336,data_seed=33550336,logging_steps=1,save_strategy='steps',save_steps=18,save_total_limit=4,report_to=['tensorboard'],remove_unused_columns=False,disable_tqdm=False)
    trainer=core.WeightedTrainer(model=model,args=args,train_dataset=data,data_collator=DataCollatorForSeq2Seq(tok,padding=True,label_pad_token_id=-100),processing_class=tok,end_weight=3)
    result=trainer.train();assert trainer.state.global_step==72
    adapter=Path(a.output)/'adapter';model.save_pretrained(adapter);tok.save_pretrained(adapter)
    (Path(a.output)/'training_metrics.json').write_text(json.dumps(result.metrics,indent=2))
if __name__=='__main__':main()
