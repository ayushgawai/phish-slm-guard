#!/usr/bin/env python3
"""LoRA fine-tune microsoft/Phi-3.5-mini-instruct. Real training gated behind --train-slm."""
import argparse
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train-slm",action="store_true"); ap.add_argument("--out",default="artifacts/slm-lora"); a=ap.parse_args()
    if not a.train_slm: print("Dry run: pass --train-slm (installs torch/transformers/peft via [slm]) to train Phi-3.5 LoRA."); return
    from datasets import Dataset; from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    from phishguard.data import generate
    rows=[{"text":f"Classify intent/family with rationale.\n{m.subject} {m.body}\nLabel: {m.family} malicious={m.label}"} for m in generate(40)]
    tok=AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct",trust_remote_code=True)
    model=get_peft_model(AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct",trust_remote_code=True),LoraConfig(r=8,lora_alpha=16,target_modules=["q_proj","v_proj"]))
    ds=Dataset.from_list(rows).map(lambda x: tok(x["text"],truncation=True,max_length=512))
    Trainer(model=model,args=TrainingArguments(output_dir=a.out,per_device_train_batch_size=2,num_train_epochs=1,report_to=[]),train_dataset=ds).train()
    model.save_pretrained(a.out)
if __name__=="__main__": main()
