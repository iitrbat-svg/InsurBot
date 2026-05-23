"""FastAPI Backend — Insurance Intel"""
import os, json, asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2","true")
os.environ.setdefault("LANGCHAIN_PROJECT","insurance-intel")

import requests as _req
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from router        import route, CORPUS
from sql_filter    import sql_filter, get_policies
from rag_retriever import retrieve, format_chunks, warmup

app = FastAPI(title="Insurance Intel API")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
GEMINI_CANDIDATES = [(k,m) for k,m in [
    (os.getenv("GEMINI_KEY_PAID"),"gemini-2.5-flash"),
    (os.getenv("GEMINI_KEY_PAID"),"gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_1"),   "gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_1"),   "gemini-2.0-flash"),
    (os.getenv("GEMINI_KEY_2"),   "gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_2"),   "gemini-2.0-flash"),
    (os.getenv("GEMINI_KEY_3"),   "gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_3"),   "gemini-2.0-flash"),
] if k]

@app.on_event("startup")
async def startup():
    await asyncio.get_event_loop().run_in_executor(None, warmup)
    print("Ready.")

class ChatRequest(BaseModel):
    message:    str
    session_id: str  = "default"
    summary:    str  = ""
    history:    list = []

def _sbh(): return {"apikey":SUPABASE_KEY,"Authorization":f"Bearer {SUPABASE_KEY}"}

def build_fc_context(policy_ids, question):
    policies = get_policies(policy_ids)
    rag      = retrieve(question, policy_ids=policy_ids, top_k=10, min_similarity=0.2)
    lines    = ["## Policy Details\n"]
    for p in policies:
        lines.append(f"### {p.get('insurer')} — {p.get('product_slug')}")
        for label,val in [
            ("Sum Insured",              f"₹{p.get('si_min_lakhs')}L–₹{p.get('si_max_lakhs')}L"),
            ("Room Rent",                p.get("room_rent_limit")),
            ("Initial Waiting",          "30 days"),
            ("PED Waiting",              f"{p.get('ped_waiting_months')} months"),
            ("Specific Illness Waiting", f"{p.get('specific_illness_waiting')} months" if p.get('specific_illness_waiting') else None),
            ("Co-payment",               f"{p.get('copayment_percent',0)}%"),
            ("Maternity",                "Yes" if p.get("maternity_covered") else "No"),
            ("Restore",                  p.get("restore_type") or ("Yes" if p.get("restore_benefit") else "No")),
            ("NCB",                      f"{p.get('no_claim_bonus_percent')}%/yr" if p.get("no_claim_bonus_percent") else None),
            ("Network Hospitals",        p.get("network_hospitals")),
            ("CSR",                      f"{p.get('claim_settlement_ratio')}%" if p.get("claim_settlement_ratio") else None),
        ]:
            if val and "None" not in str(val): lines.append(f"- **{label}**: {val}")
        lines.append("")
    lines += ["\n## Relevant Policy Clauses\n", format_chunks(rag, max_chunks=8)]
    return "\n".join(lines)

def build_rag_context(question, policy_ids, section_tags):
    return format_chunks(retrieve(question, policy_ids=policy_ids,
                                  section_tags=section_tags, top_k=8), max_chunks=6)

def _format_sql(result):
    candidates = result.get("candidates",[])
    if not candidates: return "No matching policies found."
    lines = [f"Found {len(candidates)} matching policies:\n"]
    for p in candidates:
        lines.append(
            f"- **{p.get('insurer')} — {p.get('product_slug')}**: "
            f"SI up to ₹{p.get('si_max_lakhs')}L, "
            f"Initial wait 30 days, "
            f"PED wait {p.get('ped_waiting_months')}m, "
            f"Specific illness wait {p.get('specific_illness_waiting') or 'N/A'}m, "
            f"Room rent: {p.get('room_rent_limit')}, "
            f"Co-pay: {p.get('copayment_percent',0)}%, "
            f"CSR: {p.get('claim_settlement_ratio')}%"
        )
    return "\n".join(lines)

# ── Specified disease list (shared between router and synthesizer) ─────────────
SPECIFIED_DISEASES = [
    "gall bladder","bile duct","gallstone","kidney stone","calculi","hernia",
    "cataract","knee replacement","joint replacement","tonsil","adenoid",
    "hysterectomy","benign tumor","benign tumour","varicose vein","piles",
    "fissure","fistula","hydrocele","sinusitis","spinal disc","prolapse disc",
    "pilonidal","mastoid","tympanoplasty","prostate hypertrophy",
    "gastric ulcer","duodenal ulcer","osteoarthritis","osteoporosis",
]

ACUTE_ILLNESSES = [
    "jaundice","dengue","typhoid","malaria","fever","appendicitis","fracture",
    "accident","injury","pneumonia","infection","flu","covid","viral","bacterial",
    "food poisoning","diarrhea","vomiting","stroke","heart attack","emergency",
]

def classify_condition(conditions: list) -> str:
    """Returns 'specified', 'acute', or 'ped' for the primary condition."""
    joined = " ".join(conditions).lower()
    if any(d in joined for d in SPECIFIED_DISEASES): return "specified"
    if any(d in joined for d in ACUTE_ILLNESSES):    return "acute"
    return "ped"

BASE = """You are an expert Indian health insurance advisor.
Answer based ONLY on the policy information provided.
Always cite specific numbers: waiting periods in months, limits in ₹, percentages.
Never guess coverage. If genuinely not in context, say "Not found in retrieved policy data."
Use clean formatting with bullet points and bold for key numbers."""

INTENT_PROMPTS = {

"single_policy_detail": """
Answer precisely. Quote exact clause text for waiting periods, limits, exclusions.
If condition is an acute illness (jaundice, dengue, typhoid, fever etc.) — state clearly:
"Covered after 30-day initial waiting period as a general hospitalization claim."
If condition is a specified disease (gall bladder, hernia, cataract etc.) — state the
exact waiting period from the policy wording.
Never say "not found" for standard acute illness coverage — all policies cover it after 30 days.""",

"comparison": """
Compare side by side. One section per parameter.
For each policy state the exact value. End with a summary:
Policy | Parameter1 | Parameter2 | Verdict
Never say "covered" without evidence from the retrieved text.""",

"recommendation": """
Recommend policies ranked for the user's profile.

CONDITION CLASSIFICATION — apply before answering:
ACUTE ILLNESS (jaundice, dengue, typhoid, fever, appendicitis, fracture etc.):
  → Covered by ALL policies after standard 30-day initial waiting period.
  → State this upfront, then rank by overall value: CSR, network size, room rent, premium.

SPECIFIED DISEASE (gall bladder, hernia, cataract, knee replacement, kidney stones etc.):
  → Mandatory 24-month wait in most policies per IRDAI guidelines.
  → State the waiting period per policy. Rank by shortest wait, then overall value.

PRE-EXISTING DISEASE (diabetes, hypertension, cardiac, declared at buying):
  → PED waiting period applies (24–48 months depending on policy).
  → Rank by shortest PED wait, then overall value.

For each recommended policy state:
1. How the condition is handled (waiting period / coverage from day 31)
2. Cashless network size
3. Premium range if available
4. Key benefit and key concern
End with a clear top pick with one-line reason.""",

"condition_specific": """
FIRST: Classify the condition:

ACUTE ILLNESS (jaundice, dengue, typhoid, fever, appendicitis, fracture, emergency etc.):
State upfront: "This is an acute illness covered under general hospitalization by all
Indian health insurance policies after the standard 30-day initial waiting period.
No specific waiting period applies beyond the first 30 days."
Then for each policy confirm: cashless network, room rent limit, co-payment if any.

SPECIFIED DISEASE (gall bladder, hernia, cataract, kidney stones, knee replacement etc.):
State upfront: "This condition is listed under Specified Disease Waiting Period.
Most policies require 24 months waiting. Here are the specifics per policy:"
Then list each policy's specific illness waiting period.

PRE-EXISTING DISEASE (diabetes, hypertension, cardiac — declared conditions):
List PED waiting period per policy.

Never apply specified disease logic to acute illnesses. Never say "not found"
for standard acute illness hospitalization coverage.""",

"premium_lookup": """
State premium clearly: age band, sum insured, zone, figure.
Note zone variation (A=metro higher, C=rest lower) and loading for conditions.""",

"eligibility_check": """
State min/max entry age for each policy. Flag any where the user's age is outside range.
List all eligible policies clearly.""",

"evaluation": """
Evaluate across 7 dimensions:
1. Initial waiting (30 days standard)
2. Specific illness waiting period
3. PED waiting period
4. Room rent limit
5. Co-payment
6. Restore/NCB
7. Network hospitals + CSR
If user has a condition, classify it (acute/specified/PED) and evaluate accordingly.
Verdict: Recommended / Conditional / Not Recommended + one-line reason.""",

"claims_process": """
Step by step. Cashless: intimation → pre-auth → treatment → discharge.
Reimbursement: treatment → collect documents → submit → settlement.
Include timelines and document list from the policy context.""",

"corpus_search": """
List all matching policies with the specific clause or value that answers the query.
Exclude policies where data doesn't confirm the feature — never assume.""",

"general_knowledge": """
Explain clearly in simple language with a practical Indian insurance example.
Under 150 words unless complexity requires more.""",

"followup": """
Answer using prior context. Be concise.
If referring to a specific policy discussed earlier, use only that policy's data.""",
}

def build_system_prompt(intent, conditions=None):
    prompt = BASE + INTENT_PROMPTS.get(intent,"")
    # Inject condition classification hint for condition-related intents
    if conditions and intent in ("recommendation","condition_specific","single_policy_detail"):
        ctype = classify_condition(conditions)
        hints = {
            "acute":     "\n\nCONDITION TYPE: ACUTE ILLNESS — covered after 30-day initial wait. No specified disease waiting period.",
            "specified": "\n\nCONDITION TYPE: SPECIFIED DISEASE — 24-month waiting period typically applies.",
            "ped":       "\n\nCONDITION TYPE: PRE-EXISTING DISEASE — PED waiting period applies (24-48 months).",
        }
        prompt += hints.get(ctype,"")
    return prompt

# ── Pipeline ───────────────────────────────────────────────────────────────────

async def process_chat(req: ChatRequest):
    router_result  = route(req.message, summary=req.summary, conversation_history=req.history)
    intent         = router_result["intent"]
    retrieval_mode = router_result["retrieval_mode"]
    policies       = router_result["policies_mentioned"]
    filters        = router_result["filters"]
    section_tags   = router_result["section_tags"]
    question       = router_result["specific_question"] or req.message
    conditions     = filters.get("conditions",[])

    yield f"data: {json.dumps({'type':'route','intent':intent,'retrieval_mode':retrieval_mode,'policies':policies})}\n\n"

    if router_result.get("needs_clarification"):
        q = router_result.get("clarification_question","Could you provide more details?")
        yield f"data: {json.dumps({'type':'answer','text':q})}\n\n"
        yield "data: [DONE]\n\n"; return

    if intent == "missing_docs":
        names = router_result.get("not_in_corpus",[])
        msg   = f"I don't have {', '.join(names)} in my database. Upload its policy document and I'll analyse it."
        yield f"data: {json.dumps({'type':'answer','text':msg})}\n\n"
        yield "data: [DONE]\n\n"; return

    # For acute illnesses, override section_tags to search general coverage
    if conditions and classify_condition(conditions) == "acute":
        section_tags = ["general_coverage","claims_process"]
        question     = (f"What is covered under general hospitalization and acute illness? "
                        f"What is the initial waiting period and cashless facility details?")

    context = ""
    if retrieval_mode == "llm_only":
        context = ""
    elif retrieval_mode == "sql_only":
        res     = sql_filter(filters, intent)
        context = _format_sql(res)
    elif retrieval_mode == "full_context":
        context = build_fc_context(policies, question) if policies else build_rag_context(question, None, section_tags or None)
    else:
        candidate_ids = None
        if intent in ("recommendation","condition_specific","corpus_search","evaluation"):
            sql_res       = sql_filter(filters, intent, limit=12)
            candidate_ids = [c["id"] for c in sql_res["candidates"]]
            yield f"data: {json.dumps({'type':'trace','text':f'SQL filtered to {len(candidate_ids)} candidates'})}\n\n"
        search_ids = policies if policies else candidate_ids
        context    = build_rag_context(question, search_ids, section_tags or None)

    yield f"data: {json.dumps({'type':'trace','text':'Synthesizing...'})}\n\n"

    system       = build_system_prompt(intent, conditions)
    profile      = {k:v for k,v in filters.items() if v is not None}
    user_content = f"Question: {req.message}\n"
    if profile:      user_content += f"User profile: {json.dumps(profile)}\n"
    if context:      user_content += f"\nPolicy Information:\n{context}"

    messages = [SystemMessage(content=system), HumanMessage(content=user_content)]

    last_err = None
    for key, model_name in GEMINI_CANDIDATES:
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=key,
                                          temperature=0.3, streaming=True)
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield f"data: {json.dumps({'type':'answer','text':chunk.content})}\n\n"
            yield "data: [DONE]\n\n"; return
        except Exception as e:
            if any(x in str(e) for x in ["429","quota","503","504","Deadline"]):
                last_err = e; continue
            raise

    yield f"data: {json.dumps({'type':'error','text':f'All models failed: {last_err}'})}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(process_chat(req), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/health")
def health(): return {"status":"ok","policies":len(CORPUS)}

@app.get("/policies")
def list_policies(): return [{"id":p["id"],"insurer":p["insurer"],"slug":p["product_slug"]} for p in CORPUS]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
