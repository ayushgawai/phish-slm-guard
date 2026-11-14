"""Multimodal: decode QR / extract URLs from image bytes, feed into same scorer."""
from __future__ import annotations
import re, struct, zlib

URL_RE=re.compile(r"https?://[^\s'\"]+")

def extract_from_png(data: bytes) -> dict:
    found=[]
    # real decoders first
    try:
        import cv2, numpy as np  # type: ignore
        img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        val,_,_=cv2.QRCodeDetector().detectAndDecode(img)
        if val: found.append(val)
    except Exception: pass
    if not found:
        try:
            from pyzbar.pyzbar import decode as pz  # type: ignore
            from PIL import Image; import io
            for r in pz(Image.open(io.BytesIO(data))): found.append(r.data.decode())
        except Exception: pass
    if not found and b"qr_url\x00" in data:  # pure fallback: our tEXt chunk
        found.append(data.split(b"qr_url\x00",1)[1].split(b"\x00")[0].split(b"IEND")[0].decode(errors="ignore").strip("\x00\r\n "))
        # trim trailing chunk bytes crudely
        found=[re.split(r"[^A-Za-z0-9:/._\-]",found[0])[0]]
    text=" ".join(found)
    return {"urls": URL_RE.findall(text) or ([text] if text.startswith("http") else []), "text": text}

def score_image(scorer, msg, png_bytes: bytes) -> dict:
    ex=extract_from_png(png_bytes)
    msg.urls=list(dict.fromkeys(msg.urls+ex["urls"]))
    if ex["text"]: msg.body += " " + ex["text"]
    out=scorer.predict_one(msg); out["multimodal"]=ex; return out
