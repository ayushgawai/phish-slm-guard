#!/usr/bin/env python3
import argparse, os, sys
sys.path.insert(0,"src")
from phishguard.data import generate, write_jsonl
from phishguard.model import FallbackScorer
from phishguard.distill import distill
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="artifacts/student"); a=ap.parse_args()
    msgs=generate(60,seed=1); os.makedirs("data",exist_ok=True); write_jsonl(msgs,"data/train.jsonl"); write_jsonl(generate(40,seed=99,adversarial=True),"data/adversarial.jsonl")
    teacher=FallbackScorer().fit(msgs); distill(teacher,generate(60,seed=2),a.out); print(f"student -> {a.out}")
