"""Metrics + evaluation harness + markdown report + offline A/B with bootstrap CI."""
from __future__ import annotations
import argparse, math, os, random
from collections import defaultdict
from .data import FAMILIES, MALICIOUS, generate
from .model import FallbackScorer

def auc(y,s):
    pos=[v for v,t in zip(s,y) if t]; neg=[v for v,t in zip(s,y) if not t]
    if not pos or not neg: return 0.0
    wins=sum(1 if a>b else 0.5 if a==b else 0 for a in pos for b in neg); return wins/(len(pos)*len(neg))
def ece(y,s,bins=10):
    n=len(y); out=0.0
    for i in range(bins):
        lo,hi=i/bins,(i+1)/bins; idx=[j for j,v in enumerate(s) if lo<=v<hi or (i==bins-1 and v==hi)]
        if idx: out+=len(idx)/n*abs(sum(y[j] for j in idx)/len(idx)-sum(s[j] for j in idx)/len(idx))
    return out
def precision_at_fpr(y,s,target=0.001):
    order=sorted(range(len(y)),key=lambda i:-s[i]); best=0.0
    for k in range(1,len(order)+1):
        sel=order[:k]; fp=sum(1 for j in sel if not y[j]); neg=sum(1 for v in y if not v)
        if neg and fp/neg<=target: best=sum(y[j] for j in sel)/k
    return best

def evaluate(scorer, msgs, threshold=0.5):
    rows=[(m,scorer.predict_one(m)) for m in msgs]; y=[m.label for m,_ in rows]; s=[r["probability"] for _,r in rows]
    pred=[int(v>=threshold) for v in s]
    per={}
    for fam in FAMILIES:
        tp=sum(1 for m,p in zip(msgs,pred) if m.family==fam and p==m.label and p==1)
        # per-family treated one-vs-rest for malicious families; benign = accuracy as negative
        idx=[i for i,m in enumerate(msgs) if m.family==fam]
        if fam in MALICIOUS:
            tp=sum(1 for i in idx if pred[i]==1); fn=len(idx)-tp; fp=sum(1 for i,m in enumerate(msgs) if m.family!=fam and pred[i]==1 and False)
            per[fam]={"precision":round(tp/max(1,tp+sum(1 for i,m in enumerate(msgs) if m.family==fam and False)),4) if False else None,
                      "recall":round(tp/max(1,len(idx)),4),"support":len(idx)}
            # binary precision overall computed below; per-family precision = frac of predicted-this-family correct
            pf=[r for m,r in rows if r["family"]==fam]; per[fam]["precision"]=round(sum(1 for m,r in rows if r["family"]==fam and m.family==fam)/max(1,len(pf)),4)
        else:
            per[fam]={"precision":None,"recall":round(sum(1 for i in idx if pred[i]==0)/max(1,len(idx)),4),"support":len(idx)}
    tp=sum(1 for a,b in zip(y,pred) if a and b); fp=sum(1 for a,b in zip(y,pred) if not a and b); fn=sum(1 for a,b in zip(y,pred) if a and not b)
    fps=[{"id":m.id,"family":m.family,"prob":r["probability"],"signals":r["signals"]} for m,r in rows if not m.label and r["probability"]>=threshold][:10]
    fns=[{"id":m.id,"family":m.family,"prob":r["probability"],"signals":r["signals"]} for m,r in rows if m.label and r["probability"]<threshold][:10]
    return {"n":len(msgs),"precision":round(tp/max(1,tp+fp),4),"recall":round(tp/max(1,tp+fn),4),
            "roc_auc":round(auc(y,s),4),"ece":round(ece(y,s),4),"precision_at_fpr_0.1pct":round(precision_at_fpr(y,s),4),
            "per_family":per,"top_false_positives":fps,"top_false_negatives":fns}

def bootstrap_diff(a,b,msgs,n=200,seed=0):
    """A/B: difference in recall (B-A) with bootstrap 95% CI."""
    rng=random.Random(seed); diffs=[]
    for _ in range(n):
        idx=[rng.randrange(len(msgs)) for _ in msgs]
        def rec(sc):
            tp=sum(1 for i in idx if msgs[i].label and sc.predict_one(msgs[i])["probability"]>=0.5); pos=sum(1 for i in idx if msgs[i].label)
            return tp/max(1,pos)
        diffs.append(rec(b)-rec(a))
    diffs.sort(); return {"recall_diff_b_minus_a":round(sum(diffs)/len(diffs),4),"ci95":[round(diffs[int(.025*len(diffs))],4),round(diffs[int(.975*len(diffs))],4)]}

def run_report(out="reports/eval_report.md"):
    train=generate(60,seed=1); scorer=FallbackScorer().fit(train)
    test=generate(40,seed=99); adv=generate(40,seed=99,adversarial=True)
    half=sorted(test,key=lambda m:m.timestamp); drift={"early":evaluate(scorer,half[:len(half)//2]),"late":evaluate(scorer,half[len(half)//2:])}
    res={"standard":evaluate(scorer,test),"adversarial":evaluate(scorer,adv),"time_split_drift":drift}
    # A/B: student (distilled, see distill.py) vs fallback — here threshold variant as v2 proxy is replaced by student in scripts
    os.makedirs(os.path.dirname(out) or ".",exist_ok=True)
    with open(out,"w") as f:
        f.write("# Evaluation Report\n\n")
        for k,v in [("Standard",res["standard"]),("Adversarial (LLM-rewritten paraphrase)",res["adversarial"])]:
            f.write(f"## {k}\n\n| Metric | Value |\n|---|---|\n")
            for mk in ["n","precision","recall","roc_auc","ece","precision_at_fpr_0.1pct"]: f.write(f"| {mk} | {v[mk]} |\n")
            f.write("\n| Family | Precision | Recall | Support |\n|---|---|---|---|\n")
            for fam,d in v["per_family"].items(): f.write(f"| {fam} | {d['precision']} | {d['recall']} | {d['support']} |\n")
            f.write(f"\nTop FP: `{v['top_false_positives'][:3]}`\n\nTop FN: `{v['top_false_negatives'][:3]}`\n\n")
        f.write(f"## Time-split drift\n\nEarly recall={drift['early']['recall']} AUC={drift['early']['roc_auc']} · Late recall={drift['late']['recall']} AUC={drift['late']['roc_auc']}\n")
    return res

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="reports/eval_report.md"); a=ap.parse_args()
    print(run_report(a.out))
if __name__=="__main__": main()
