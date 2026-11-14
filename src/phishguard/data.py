"""Synthetic seeded corpus: email + Teams-style chat across attack families."""
from __future__ import annotations
import base64, io, json, random, struct, zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

FAMILIES = ["credential_phish","bec_invoice","gift_card_scam","callback_scam","qr_phish",
            "benign_marketing","benign_internal"]
MALICIOUS = FAMILIES[:5]

@dataclass
class Message:
    id: str; channel: str; family: str; label: int  # 1=malicious
    timestamp: str; sender: str; sender_domain: str; reply_to: str
    spf: str; dkim: str; subject: str; body: str
    urls: list[str] = field(default_factory=list); attachments: list[str] = field(default_factory=list)
    qr_url: str | None = None; split: str = "standard"
    def to_dict(self): return asdict(self)
    def text(self): return f"{self.subject} {self.body} {self.sender} {' '.join(self.urls)}"

_T = {
 "credential_phish": [("Urgent: verify your {brand} account","Your {brand} account will be suspended in 24 hours. Verify your password immediately at the link or lose access."),
   ("Action required: unusual sign-in","We detected unusual sign-in activity. Confirm your credentials now to restore secure access.")],
 "bec_invoice": [("Invoice {n} overdue - wire update","Hi, please process invoice {n} today and wire payment to the updated bank account. Keep this confidential, the CEO requested it."),
   ("Updated remittance details","Please change the payment details for vendor invoice {n} and transfer funds urgently. Do not call, reply by email only.")],
 "gift_card_scam": [("Quick favour needed","Are you at your desk? Purchase gift cards urgently for a client reward and send me the redemption codes. Keep this discreet."),
   ("Urgent request","I need you to buy gift cards now and email the scratch codes. I will reimburse you, please do not mention this.")],
 "callback_scam": [("Subscription renewal charge","Your antivirus subscription renewed for $499. If this was not you, call our support number immediately to cancel and get a refund."),
   ("Order confirmation - call us","A charge of $389 was placed. Call the toll free number now if you did not authorize this purchase.")],
 "qr_phish": [("Scan to review document","Please scan the QR code in the attached image to review the secure shared document and sign in."),
   ("MFA reset required","Scan the QR code attached to reset your multi factor authentication before it expires.")],
 "benign_marketing": [("Spring sale: 20% off","Explore our spring sale with discounts on selected products. Unsubscribe anytime. Shop our official store today."),
   ("Your weekly newsletter","Read our latest product updates, customer stories and upcoming webinar announcements in this newsletter.")],
 "benign_internal": [("Team standup notes","Standup notes are in the shared document. Please review the sprint board and update your tickets before Friday."),
   ("HR reminder: benefits enrolment","Benefits enrolment closes next week. Visit the HR portal via the intranet bookmark if you need to update selections.")],
}
_BRANDS=["Microsoft 365","Okta","DocuSign","PayPal"]
_SYN={"urgent":"pressing","immediately":"right away","verify":"confirm","password":"passcode",
      "account":"profile","purchase":"buy","please":"kindly","now":"today","suspended":"paused"}

def paraphrase(text: str) -> str:
    out=[]
    for w in text.split():
        k=w.lower().strip(".,")
        out.append(_SYN.get(k,w))
    return " ".join(out)

def _png_with_text(text: str) -> bytes:
    """Minimal valid PNG embedding text in a tEXt chunk (pure-stdlib QR fallback payload)."""
    def chunk(t,data):
        c=t+data; return struct.pack(">I",len(data))+c+struct.pack(">I",zlib.crc32(c)&0xffffffff)
    ihdr=struct.pack(">IIBBBBB",1,1,8,2,0,0,0)
    raw=b"\x00\xff\xff\xff"
    return (b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",ihdr)+chunk(b"tEXt",f"qr_url\x00{text}".encode())
            +chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b""))

def make_qr_png(url: str) -> bytes:
    try:
        import qrcode, io as _io  # type: ignore
        buf=_io.BytesIO(); qrcode.make(url).save(buf,format="PNG"); return buf.getvalue()
    except Exception:
        return _png_with_text(url)  # fallback still decodes via multimodal.extract_from_png

def generate(n_per_family: int = 40, seed: int = 42, adversarial: bool = False) -> list[Message]:
    rng=random.Random(seed); msgs=[]; base=datetime(2026,1,1,tzinfo=timezone.utc); i=0
    for fam in FAMILIES:
        for k in range(n_per_family):
            i+=1; mal=fam in MALICIOUS; subj_t,body_t=rng.choice(_T[fam])
            brand=rng.choice(_BRANDS); n=rng.randint(1000,9999)
            subj=subj_t.format(brand=brand,n=n); body=body_t.format(brand=brand,n=n)
            if adversarial and mal: body=paraphrase(body); subj=paraphrase(subj)
            dom = (f"{brand.split()[0].lower()}-secure-login.example.com" if fam=="credential_phish"
                   else "evil-vendor-payments.example.com" if mal else "company.example.com")
            url = f"https://{dom}/login/{n}" if mal else "https://company.example.com/portal"
            qr_url = url if fam=="qr_phish" else None
            sender = f"alerts@{dom}" if mal else f"team@company.example.com"
            msgs.append(Message(id=f"m{i:04d}", channel=rng.choice(["email","teams_chat"]), family=fam,
                label=int(mal), timestamp=(base+timedelta(hours=i*7)).isoformat(),
                sender=sender, sender_domain=dom, reply_to=(f"reply@{dom}.evil.example.com" if mal and k%3==0 else sender),
                spf=("fail" if mal and k%2==0 else "pass"), dkim=("fail" if mal and k%3==0 else "pass"),
                subject=subj, body=body, urls=[url] if fam!="qr_phish" else [],
                attachments=["qr.png"] if fam=="qr_phish" else [], qr_url=qr_url,
                split="adversarial" if adversarial else "standard"))
    rng.shuffle(msgs)
    for j,m in enumerate(msgs): m.timestamp=(base+timedelta(hours=j*7)).isoformat()  # time-split not family-correlated
    return msgs

def write_jsonl(msgs, path):
    with open(path,"w") as f:
        for m in msgs: f.write(json.dumps(m.to_dict())+"\n")

def qr_png_b64(msg: Message) -> str | None:
    return base64.b64encode(make_qr_png(msg.qr_url)).decode() if msg.qr_url else None
