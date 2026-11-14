"""Distillation: SLM/fallback teacher -> compact student; ONNX export when available."""
from __future__ import annotations
import json, os
from .model import FallbackScorer

def distill(teacher, msgs, out_dir="artifacts/student"):
    pseudo=[]
    for m in msgs:
        r=teacher.predict_one(m); m2=type(m)(**{**m.to_dict(),"label":int(r["probability"]>=0.5),"family":r["family"]}); pseudo.append(m2)
    student=FallbackScorer().fit(pseudo); os.makedirs(out_dir,exist_ok=True); student.save(f"{out_dir}/student.json")
    onnx_path=None
    try:  # best-effort real ONNX when sklearn+skl2onnx present
        if student._sk:
            from skl2onnx import to_onnx  # type: ignore
            onnx_path=f"{out_dir}/student.onnx"; open(onnx_path,"wb").write(to_onnx(student._sk[0], [""], options={"zipmap":False}).SerializeToString())
    except Exception: pass
    json.dump({"onnx":onnx_path,"note":"JSON weights are the portable student; ONNX exported when skl2onnx available"},open(f"{out_dir}/manifest.json","w"))
    return student
