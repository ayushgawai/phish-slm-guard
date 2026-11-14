"""Fallback scorer: TF-IDF + logistic regression (sklearn if present, else pure-stdlib SGD).

Same interface as the SLM: predict_one(msg) -> verdict/family/calibrated prob/signals/rationale.
"""
from __future__ import annotations
import json, math, re
from collections import Counter
from .data import FAMILIES, Message

TOK=re.compile(r"[a-z0-9]+")
def _feats(m: Message) -> list[str]:
    f=[f"spf_{m.spf}",f"dkim_{m.dkim}"]
    if m.reply_to.split("@")[-1]!=m.sender_domain: f.append("reply_to_mismatch")
    if any(d not in m.sender_domain for d in [u.split("/")[2] if "://" in u else u for u in m.urls]): f.append("url_domain_mismatch")
    if m.attachments: f.append("has_attachment")
    return f
def _text(m): return (m.subject+" "+m.body+" "+" ".join(m.urls)).lower()

class FallbackScorer:
    def __init__(self): self._sk=None; self.vocab={}; self.idf={}; self.w={}; self.b=0.0; self.fam_w={}
    def fit(self, msgs: list[Message]):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.pipeline import make_pipeline  # type: ignore
            X=[_text(m)+" "+" ".join(_feats(m)) for m in msgs]; y=[m.label for m in msgs]
            fam=make_pipeline(TfidfVectorizer(ngram_range=(1,2)),LogisticRegression(max_iter=1000))
            fam.fit(X,[m.family for m in msgs])
            binm=make_pipeline(TfidfVectorizer(ngram_range=(1,2)),LogisticRegression(max_iter=1000))
            binm.fit(X,y); self._sk=(binm,fam); return self
        except Exception:
            pass
        # pure-stdlib TF-IDF + SGD logistic (binary + per-family one-vs-rest)
        docs=[Counter(TOK.findall(_text(m)))+Counter(_feats(m)) for m in msgs]
        df=Counter()
        for d in docs:
            for t in d: df[t]+=1
        self.vocab={t:i for i,t in enumerate(sorted(df))}
        n=len(docs); self.idf={t:math.log((n+1)/(c+1))+1 for t,c in df.items()}
        def vec(d): return {t:(1+math.log(c))*self.idf[t] for t,c in d.items() if t in self.idf}
        vecs=[vec(d) for d in docs]
        def train(labels, epochs=300):
            w=Counter(); b=0.0
            for _ in range(epochs):
                for x,y in zip(vecs,labels):
                    z=b+sum(w[t]*v for t,v in x.items()); p=1/(1+math.exp(-max(-30,min(30,z))))
                    for t,v in x.items(): w[t]-=0.1*(p-y)*v
                    b-=0.1*(p-y)
            return w,b
        self.w,self.b=train([m.label for m in msgs])
        for fam in FAMILIES: self.fam_w[fam]=train([int(m.family==fam) for m in msgs],epochs=120)
        self._docs=docs; return self
    def _prob(self, m):
        if self._sk:
            binm,_=self._sk; x=[_text(m)+" "+" ".join(_feats(m))]
            return float(binm.predict_proba(x)[0][1]), {}
        d=Counter(TOK.findall(_text(m)))+Counter(_feats(m))
        x={t:(1+math.log(c))*self.idf.get(t,1) for t,c in d.items() if t in self.vocab}
        z=self.b+sum(self.w.get(t,0)*v for t,v in x.items()); p=1/(1+math.exp(-max(-30,min(30,z))))
        contrib=sorted(((self.w.get(t,0)*v,t) for t,v in x.items()),reverse=True)
        return p, contrib
    def predict_one(self, m: Message) -> dict:
        p,contrib=self._prob(m)
        if self._sk:
            _,fam=self._sk; family=str(fam.predict([_text(m)+" "+" ".join(_feats(m))])[0])
            signals=_feats(m)
        else:
            d=Counter(TOK.findall(_text(m)))+Counter(_feats(m))
            x={t:(1+math.log(c))*self.idf.get(t,1) for t,c in d.items() if t in self.vocab}
            scores={f:b+sum(w.get(t,0)*v for t,v in x.items()) for f,(w,b) in self.fam_w.items()}
            family=max(scores,key=scores.get) if p>=0.5 else max((f for f in scores if f.startswith("benign")),key=scores.get)
            signals=[t for _,t in contrib[:5] if _>0]+_feats(m)
        verdict="malicious" if p>=0.5 else "benign"
        if verdict=="benign": family=family if family.startswith("benign") else "benign_internal"
        return {"verdict":verdict,"family":family,"probability":round(float(p),4),
                "signals":signals[:8],"rationale":f"{verdict}: top signals {', '.join(signals[:3]) or 'none'}"}
    def save(self,path):
        # ponytail: JSON weights for stdlib model; sklearn path re-trains quickly, not persisted here
        with open(path,"w") as f: json.dump({"vocab":self.vocab,"idf":self.idf,"w":dict(self.w),"b":self.b,
            "fam_w":{k:[dict(w),b] for k,(w,b) in self.fam_w.items()}},f)
    @classmethod
    def load(cls,path):
        s=cls(); d=json.load(open(path)); s.vocab=d["vocab"]; s.idf=d["idf"]; s.w=d["w"]; s.b=d["b"]
        s.fam_w={k:(Counter(w),b) for k,(w,b) in d["fam_w"].items()}; return s

# SLM interface stub (real Phi-3.5 LoRA in scripts/train_slm.py, gated by --train-slm)
class SLMScorer:
    def __init__(self, model_id="microsoft/Phi-3.5-mini-instruct"): self.model_id=model_id
    def predict_one(self, m): raise RuntimeError("SLM not loaded; use FallbackScorer or run scripts/train_slm.py --train-slm")
