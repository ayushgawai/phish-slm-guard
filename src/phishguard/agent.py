"""Investigation agent: tool loop, deterministic offline mock LLM, guardrails.

Guardrails: max_tool_calls cap; malicious verdict needs >=2 evidence items.
"""
from __future__ import annotations
import re
from .data import Message

MOCK_AGE={"evil-vendor-payments.example.com":3,"microsoft-secure-login.example.com":5,"okta-secure-login.example.com":4}
BRANDS=["microsoft","okta","paypal","docusign","company","google","apple"]
HOMO={"0":"o","1":"l","rn":"m","а":"a"}

def _dom(url_or_host): return url_or_host.split("/")[2] if "://" in url_or_host else url_or_host.split("/")[0]
def expand_redirect_chain(url): return {"tool":"expand_redirect_chain","url":url,"chain":[url],"final":url}
def domain_age_lookup(host):
    d=_dom(host); return {"tool":"domain_age_lookup","domain":d,"age_days":MOCK_AGE.get(d, 4000 if d.endswith("company.example.com") else 12)}
def lookalike_domain_check(host):
    d=_dom(host); base=d.split(".")[0]
    hit=[b for b in BRANDS if b in base and d not in (f"{b}.example.com",f"{b}.com")]
    dist=min((sum(1 for a,b in zip(base,x) if a!=b)+abs(len(base)-len(x)) for x in BRANDS), default=99)
    return {"tool":"lookalike_domain_check","domain":d,"lookalike_of":hit,"edit_distance":dist,"homoglyph":any(k in d for k in HOMO)}
def header_auth_check(m: Message):
    return {"tool":"header_auth_check","spf":m.spf,"dkim":m.dkim,
            "reply_to_mismatch": m.reply_to.split("@")[-1]!=m.sender_domain, "sender_domain":m.sender_domain}
def summarize_verdict(evidence, base): return {"tool":"summarize_verdict","evidence_count":len(evidence)}

TOOLS={"expand_redirect_chain":expand_redirect_chain,"domain_age_lookup":domain_age_lookup,
       "lookalike_domain_check":lookalike_domain_check,"header_auth_check":header_auth_check}

class MockLLM:
    """Deterministic offline client: plans tools from message surface, no network."""
    def plan(self, m: Message) -> list[tuple[str,object]]:
        calls=[("header_auth_check",m)]
        for u in (m.urls+([m.qr_url] if m.qr_url else [])):
            calls += [("expand_redirect_chain",u),("domain_age_lookup",u),("lookalike_domain_check",u)]
        return calls

class Agent:
    def __init__(self, scorer, llm=None, max_tool_calls=5): self.scorer=scorer; self.llm=llm or MockLLM(); self.max=max_tool_calls
    def investigate(self, m: Message) -> dict:
        base=self.scorer.predict_one(m); evidence=[]; calls=0; log=[]
        for name,arg in self.llm.plan(m):
            if calls>=self.max: break
            calls+=1; r=TOOLS[name](arg); log.append(r)
            if name=="domain_age_lookup" and r["age_days"]<30: evidence.append(f"new domain {r['domain']} ({r['age_days']}d)")
            if name=="lookalike_domain_check" and r["lookalike_of"]: evidence.append(f"lookalike of {r['lookalike_of'][0]}: {r['domain']}")
            if name=="header_auth_check" and (r["spf"]=="fail" or r["reply_to_mismatch"]): evidence.append("header auth anomaly (SPF/reply-to)")
        verdict=base["verdict"]
        if verdict=="malicious" and len(evidence)<2:  # guardrail: require evidence
            verdict="suspicious"; explanation=f"Model flagged {base['family']} (p={base['probability']}) but only {len(evidence)} evidence item(s); escalated as suspicious, not malicious."
        elif verdict=="malicious":
            explanation=f"Likely {base['family']} (p={base['probability']}): " + "; ".join(evidence) + "."
        else:
            explanation=f"Likely benign (p_malicious={base['probability']}): no corroborating malicious evidence."
        return {"verdict":verdict,"family":base["family"],"probability":base["probability"],
                "evidence":evidence,"tool_calls":calls,"tool_log":log,"explanation":explanation}
