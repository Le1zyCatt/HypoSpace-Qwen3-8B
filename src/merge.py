import argparse
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    m=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,device_map={'':0})
    m=PeftModel.from_pretrained(m,a.adapter).merge_and_unload();m.save_pretrained(a.output)
    AutoTokenizer.from_pretrained(a.model,local_files_only=True).save_pretrained(a.output)
if __name__=='__main__':main()
