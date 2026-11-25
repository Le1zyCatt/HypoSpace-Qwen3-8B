"""Weighted objective and NF4/LoRA construction extracted from the E16 training implementation.
Paths and orchestration moved to train.py; objective and model settings are preserved.
"""
import os
import torch
from torch.utils.data import SequentialSampler
from torch.utils.checkpoint import checkpoint
from transformers import Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
WEIGHTS=os.environ.get('MODEL_PATH','Qwen/Qwen3-8B')

def example_loss(logits,labels,end_weight):
 result=[]
 for i in range(logits.shape[0]):
  pred=logits[i,:-1];truth=labels[i,1:];w=(truth!=-100).float();w[(truth==151668)|(truth==151645)]=float(end_weight);total=pred.new_zeros((),dtype=torch.float32)
  for j in range(0,len(truth),128):
   if not (truth[j:j+128]!=-100).any():continue
   def ce(a,b,c):return (torch.nn.functional.cross_entropy(a.float(),b,reduction='none',ignore_index=-100)*c).sum()
   total=total+checkpoint(ce,pred[j:j+128],truth[j:j+128],w[j:j+128],use_reentrant=False)
  result.append(total/w.sum())
 return torch.stack(result).mean()

def chunked_model_loss(model,inputs,labels,end_weight):
 base=model.get_base_model() if hasattr(model,'get_base_model') else model
 out=base.model(**inputs,use_cache=False);hidden=out.last_hidden_state;head=base.get_output_embeddings();losses=[]
 for i in range(hidden.shape[0]):
  mask=labels[i,1:]!=-100;h=hidden[i,:-1][mask];truth=labels[i,1:][mask];w=torch.ones_like(truth,dtype=torch.float32);w[(truth==151668)|(truth==151645)]=float(end_weight);total=h.new_zeros((),dtype=torch.float32)
  for j in range(0,len(truth),128):
   def linear_ce(a,b,c):return (torch.nn.functional.cross_entropy(head(a).float(),b,reduction='none')*c).sum()
   total=total+checkpoint(linear_ce,h[j:j+128],truth[j:j+128],w[j:j+128],use_reentrant=False)
  losses.append(total/w.sum())
 return torch.stack(losses).mean(),out

class WeightedTrainer(Trainer):
 def __init__(self,*a,end_weight=1,**kw):super().__init__(*a,**kw);self.end_weight=end_weight;self.model_accepts_loss_kwargs=False
 def _get_train_sampler(self,train_dataset=None):return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
 def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
  labels=inputs.pop('labels')
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=labels.is_cuda):loss,out=chunked_model_loss(model,inputs,labels,self.end_weight)
  return (loss,out) if return_outputs else loss

def make_model():
 from transformers import AutoModelForCausalLM,BitsAndBytesConfig
 m=AutoModelForCausalLM.from_pretrained(WEIGHTS,local_files_only=True,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16));m=prepare_model_for_kbit_training(m,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False});m=get_peft_model(m,LoraConfig(r=32,lora_alpha=32,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']));m.config.use_cache=False;return m
