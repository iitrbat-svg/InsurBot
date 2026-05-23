"""
FastAPI Backend — Insurance Intel
"""
import os, json, asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT",    "insurance-intel")

import requests as _req
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from router        import route, CORPUS
from sql_filter    import sql_filter, get_policies
from rag_retriever import retrieve, format_chunks, warmup

app = FastAPI(title="Insurance Intel API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

GEMINI_CANDIDATES = [(k,m) for k,m in [
    (os.getenv("GEMINI_KEY_PAID"), "gemini-2.5-flash"),
    (os.getenv("GEMINI_KEY_PAID"), "gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_1"),    "gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_1"),    "gemini-2.0-flash"),
    (os.getenv("GEMINI_KEY_2"),    "gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_2"),    "gemini-2.0-flash"),
    (os.getenv("GEMINI_KEY_3"),    "gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_3"),    "gemini-2.0-flash"),
] if k]

@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, warmup)
    print("Ready.")

class ChatRequest(BaseModel):
    message:    str
    session_id: str  = "default"
    summary:    str  = ""
    history:    list = []

def _sbh():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

def build_fc_context(policy_ids, question):
    policies    = get_policies(policy_ids)
    rag         = retrieve(question, policy_ids=policy_ids, top_k=10, min_similarity=0.2)
    rag_context = format_chunks(rag, max_chunks=8)
    lines = ["## Policy Details\n"]
    for p in policies:
        lines.append(f"### {p.get('insurer')} — {p.get('product_slug')}")
        for label, val in [
            ("Sum Insured",       f"₹{p.get('si_min_lakhs')}L–₹{p.get('si_max_lakhs')}L"),
            ("Room Rent",         p.get("room_rent_limit")),
            ("PED Waiting",       f"{p.get('ped_waiting_months')} months"),
            ("Specific Illness Waiting", f"{p.get('specific_illness_waiting')} months" if p.get('specific_illness_waiting') else None),
            ("Co-payment",        f"{p.get('copayment_percent',0)}%"),
            ("Maternity",         "Yes" if p.get("maternity_covered") else "No"),
            ("Restore",           p.get("restore_type") or ("Yes" if p.get("restore_benefit") else "No")),
            ("NCB",               f"{p.get('no_claim_bonus_percent')}%/yr" if p.get("no_claim_bonus_percent") else None),
            ("Network Hospitals", p.get("network_hospitals")),
            ("Claim Settlement",  f"{p.get('claim_settlement_ratio')}%" if p.get("claim_settlement_ratio") else None),
        ]:
            if val and "None" not in str(val):
                lines.append(f"- **{label}**: {val}")
        lines.append("")
    lines += ["\n## Relevant Policy Clauses\n", rag_context]
    return "\n".join(lines)

def build_rag_context(question, policy_ids, section_tags):
    rag = retrieve(question, policy_ids=policy_ids, section_tags=section_tags, top_k=8)
    return format_chunks(rag, max_chunks=6)

def _format_sql(result, intent):
    candidates = result.get("candidates", [])
    if not candidates:
        return "No matching policies found for the given criteria."
    lines = [f"Found {len(candidates)} matching policies:\n"]
    for p in candidates:
        lines.append(
            f"- **{p.get('insurer')} — {p.get('product_slug')}**: "
            f"SI up to ₹{p.get('si_max_lakhs')}L, "
            f"PED wait {p.get('ped_waiting_months')}m, "
            f"Specific illness wait {p.get('specific_illness_waiting') or 'N/A'}m, "
            f"Room rent: {p.get('room_rent_limit')}, "
            f"Co-pay: {p.get('copayment_percent',0)}%, "
            f"CSR: {p.get('claim_settlement_ratio')}%"
        )
    return "\n".join(lines)

# ── Synthesizer prompts ────────────────────────────────────────────────────────

BASE = """You are an expert Indian health insurance advisor.
Answer based ONLY on the policy information provided in the context.
Always cite specific numbers: waiting periods in months, limits in ₹, percentages.
If information is not in the context, say explicitly: "Not found in retrieved policy data."
Never guess or assume coverage. Use clean formatting with bullet points and bold for key numbers."""

INTENT_PROMPTS = {

"single_policy_detail": """
Answer the specific question about this policy precisely.
Extract exact clause text where relevant — quote waiting periods, limits, and exclusions verbatim.
If the question involves a condition under 'specified disease waiting period' (gall bladder,
hernia, cataract, knee replacement, etc.), state the exact waiting period from the policy.
If not found in the retrieved text, say so — do not infer from general knowledge.""",

"comparison": """
Compare the policies side by side on the exact parameters asked.
Structure: one section per parameter. For each parameter state the value per policy.
IMPORTANT: If a specific condition or clause is not found in the retrieved text for a policy,
say "Not found in retrieved data" — never say "covered" without evidence.
End with a summary table: Policy | Parameter1 | Parameter2 | Parameter3.""",

"recommendation": """
You are recommending health insurance for the user's specific profile.

CRITICAL RULES:
1. SPECIFIED DISEASE WARNING: Gall bladder, hernia, cataract, kidney stones, knee replacement,
   and similar conditions fall under 'Specified Disease Waiting Period' in ALL Indian policies.
   If the user's condition is one of these, state upfront:
   "Note: [condition] has a mandatory waiting period under most Indian health policies.
   Here's the waiting period per policy:"
   Then list each policy's specific illness waiting period explicitly.
2. Never recommend a policy as "covers from year 1" for specified diseases without proof.
3. Rank policies by: shortest waiting period for the condition → best overall coverage → value.
4. For each policy state: waiting period for the condition, cashless network size, premium if known, key benefit, key concern.
5. End with a clear top recommendation with reasoning.""",

"corpus_search": """
List all policies that match the query.
For each, quote the specific clause or value from the policy text.
If a policy's data doesn't confirm the feature, exclude it — do not assume.
Be direct: "X policies match. Here are the specifics..." """,

"condition_specific": """
For EACH policy in the context, state clearly:
1. Is this condition covered? (Yes/No/After waiting period)
2. Waiting period (months) — distinguish PED wait vs Specific Disease wait
3. Any sub-limits or co-payment specific to this condition
4. Any loading (extra premium) for this condition

IMPORTANT: Many conditions like gall bladder stone, hernia, cataract are 'Specified Diseases'
with a mandatory 24-month wait across all insurers per IRDAI guidelines.
If this applies, state it clearly upfront before listing individual policies.""",

"premium_lookup": """
Provide premium information clearly.
State: age band, sum insured, zone, and the premium figure.
If exact premium not in context, state the range from the policy data.
Note that premiums vary by zone (A=metro, B=tier1, C=rest) and loading for conditions.""",

"eligibility_check": """
Answer clearly: can this person buy this policy?
State: minimum entry age, maximum entry age, any conditions on eligibility.
If age is outside the allowed range, say so explicitly.
List all eligible policies if multiple are in context.""",

"claims_process": """
Explain the claims process step by step.
Separate cashless and reimbursement clearly.
Include: intimation timeline, documents needed, TPA or in-house contact.
Keep steps numbered and actionable.""",

"evaluation": """
Evaluate this policy thoroughly for the user's profile.
Score across these dimensions:
1. Waiting periods (PED + specific disease) — shorter is better
2. Room rent limit — no_limit is best
3. Co-payment — 0% is best
4. Restore/Recharge benefit
5. NCB
6. Network hospitals and CSR
7. Sub-limits on critical treatments

If user has a specific condition, evaluate how this policy handles it explicitly.
End with verdict: Recommended / Conditionally Recommended / Not Recommended + one-line reason.""",

"claims_process": """
Step-by-step claims guide.
Cashless: intimation → pre-auth → treatment → discharge.
Reimbursement: treatment → documents → submission → settlement.
Include timelines and document checklist if in context.""",

"general_knowledge": """
Explain the insurance concept clearly in simple language.
Use a practical Indian example.
Keep it under 150 words unless the concept requires more.""",

"followup": """
Answer based on the prior conversation context.
If the follow-up refers to a specific policy discussed earlier, use that policy's data.
Be concise — the user already has context from the prior answer.""",
}

def build_system_prompt(intent):
    return BASE + INTENT_PROMPTS.get(intent, "")

# ── Pipeline ───────────────────────────────────────────────────────────────────

async def process_chat(req: ChatRequest):
    # Route
    router_result  = route(req.message, summary=req.summary, conversation_history=req.history)
    intent         = router_result["intent"]
    retrieval_mode = router_result["retrieval_mode"]
    policies       = router_result["policies_mentioned"]
    filters        = router_result["filters"]
    section_tags   = router_result["section_tags"]
    question       = router_result["specific_question"] or req.message

    yield f"data: {json.dumps({'type':'route','intent':intent,'retrieval_mode':retrieval_mode,'policies':policies})}\n\n"

    # Clarification / missing docs
    if router_result.get("needs_clarification"):
        q = router_result.get("clarification_question","Could you provide more details?")
        yield f"data: {json.dumps({'type':'answer','text':q})}\n\n"
        yield "data: [DONE]\n\n"; return

    if intent == "missing_docs":
        names = router_result.get("not_in_corpus",[])
        msg   = f"I don't have {', '.join(names)} in my database. You can upload its policy document and I'll analyse it."
        yield f"data: {json.dumps({'type':'answer','text':msg})}\n\n"
        yield "data: [DONE]\n\n"; return

    # Retrieve context
    context = ""

    if retrieval_mode == "llm_only":
        context = ""

    elif retrieval_mode == "sql_only":
        res     = sql_filter(filters, intent)
        context = _format_sql(res, intent)

    elif retrieval_mode == "full_context":
        context = build_fc_context(policies, question) if policies else build_rag_context(question, None, section_tags or None)

    else:  # rag_filtered
        candidate_ids = None
        if intent in ("recommendation","condition_specific","corpus_search","evaluation"):
            sql_res       = sql_filter(filters, intent, limit=12)
            candidate_ids = [c["id"] for c in sql_res["candidates"]]
            yield f"data: {json.dumps({'type':'trace','text':f'SQL filtered to {len(candidate_ids)} candidates'})}\n\n"

        search_ids = policies if policies else candidate_ids
        context    = build_rag_context(question, search_ids, section_tags or None)

    yield f"data: {json.dumps({'type':'trace','text':'Synthesizing...'})}\n\n"

    # Synthesize + stream
    system = build_system_prompt(intent)
    user_content = f"Question: {req.message}\n"
    profile = {k:v for k,v in filters.items() if v is not None}
    if profile:
        user_content += f"User profile: {json.dumps(profile)}\n"
    if context:
        user_content += f"\nPolicy Information:\n{context}"

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

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(process_chat(req), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/health")
def health():
    return {"status":"ok","policies":len(CORPUS)}

@app.get("/policies")
def list_policies():
    return [{"id":p["id"],"insurer":p["insurer"],"slug":p["product_slug"]} for p in CORPUS]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
