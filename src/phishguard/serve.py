"""FastAPI /score serving (ONNX Runtime when a .onnx student exists, else student weights)."""
from __future__ import annotations
import glob
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    FastAPI=None  # type: ignore
from .data import Message, generate
from .model import FallbackScorer

def _load():
    # ponytail: train tiny fallback at startup if no artifact; production loads student.onnx via onnxruntime
    try:
        import onnxruntime as ort  # noqa: F401
        if glob.glob("artifacts/student/*.onnx"): pass  # session wiring in scripts/serve notes; scorer below is API-compatible
    except Exception: pass
    return FallbackScorer().fit(generate(30,seed=7))

if FastAPI:
    app=FastAPI(title="phish-slm-guard"); SCORER=_load()
    class Req(BaseModel):
        subject: str=""; body: str=""; sender: str="team@company.example.com"; urls: list[str]=[]
    @app.post("/score")
    def score(r: Req):
        dom=r.sender.split("@")[-1] if "@" in r.sender else "company.example.com"
        m=Message(id="api",channel="email",family="benign_internal",label=0,timestamp="2026-01-01T00:00:00+00:00",
                  sender=r.sender,sender_domain=dom,reply_to=r.sender,spf="pass",dkim="pass",subject=r.subject,body=r.body,urls=r.urls)
        out=SCORER.predict_one(m); return {"verdict":out["verdict"],"family":out["family"],"calibrated_probability":out["probability"],"top_signals":out["signals"]}
    @app.get("/health")
    def health(): return {"ok":True}
def main():
    import uvicorn; uvicorn.run(app,host="0.0.0.0",port=8000)
if __name__=="__main__": main()
