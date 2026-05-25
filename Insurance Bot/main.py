"""FastAPI Backend — Insurance Intel (with reranking, query expansion, SQL fallback)"""
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
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from router        import route, CORPUS, CORPUS_LIST
from sql_filter    import sql_filter, get_policies
from rag_retriever import retrieve, format_chunks, warmup
from memory        import (load_session, save_session, add_turn, update_profile,
                           enrich_filters, resolve_followup, should_summarize, summarize)

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
    session_id: str = "default"

def _sbh(): return {"apikey":SUPABASE_KEY,"Authorization":f"Bearer {SUPABASE_KEY}"}

# ── Query expansion ────────────────────────────────────────────────────────────
# Expands query with synonyms before embedding — improves RAG recall.
# Uses cheapest model. Falls back to original query on failure.

def expand_query(question: str, conditions: list) -> str:
    """
    Add insurance-specific synonyms and alternate phrasings to the query.
    E.g. "jaundice" → also includes "liver disease, hepatitis, acute illness hospitalization"
    This improves vector search recall without changing the meaning.
    """
    if not conditions:
        return question
    key, model = GEMINI_CANDIDATES[-1]   # cheapest/fastest model
    try:
        llm  = ChatGoogleGenerativeAI(model=model, google_api_key=key,
                                      temperature=0, max_output_tokens=150)
        resp = llm.invoke([HumanMessage(content=
            f"""For this Indian health insurance query, add 3-5 relevant medical/insurance synonyms
and alternate terms that would appear in policy documents. Keep it concise.
Output ONLY the expanded query, no explanation.

Original: {question}
Conditions mentioned: {', '.join(conditions)}

Example: "jaundice hospitalization" → "jaundice hepatitis liver disease acute illness 
hospitalization general inpatient coverage 30 day waiting period"

Expanded query:""")])
        expanded = resp.content.strip()
        return expanded if expanded else question
    except Exception:
        return question   # silent fallback

# ── Reranking ─────────────────────────────────────────────────────────────────
# After RAG retrieval, score chunks for relevance to the specific question.
# Drops weak chunks — stops irrelevant text confusing the synthesizer.

RERANK_THRESHOLD = 0.42

def rerank_chunks(chunks: list, question: str) -> list:
    """
    Filter chunks by similarity threshold.
    If too few remain above threshold, keep top-3 minimum.
    Also deduplicate near-identical chunks (same policy, similarity diff < 0.02).
    """
    if not chunks:
        return chunks

    # Apply threshold
    filtered = [c for c in chunks if c.get("similarity",0) >= RERANK_THRESHOLD]

    # Keep at least 3
    if len(filtered) < 3:
        filtered = sorted(chunks, key=lambda c: c.get("similarity",0), reverse=True)[:3]

    # Deduplicate: remove chunks from same policy+section that are near-duplicates
    seen     = {}
    deduped  = []
    for c in filtered:
        key = f"{c.get('policy_id')}_{c.get('section_tag')}"
        prev_sim = seen.get(key)
        if prev_sim is None or abs(c.get("similarity",0) - prev_sim) > 0.05:
            deduped.append(c)
            seen[key] = c.get("similarity",0)

    return deduped

# ── SQL context fallback ───────────────────────────────────────────────────────
# If RAG returns weak results, fall back to structured policy data.
# Always gives correct metadata even when clause text isn't found.

def sql_context_for_candidates(candidate_ids: list) -> str:
    """Format structured policy data as readable context."""
    if not candidate_ids: return ""
    policies = get_policies(candidate_ids[:8])
    if not policies: return ""
    lines = ["## Policy Structured Data (from database)\n"]
    for p in policies:
        lines.append(f"### {p.get('insurer')} — {p.get('product_slug')}")
        for label, val in [
            ("SI Range",                f"₹{p.get('si_min_lakhs')}L – ₹{p.get('si_max_lakhs')}L"),
            ("Room Rent",               p.get("room_rent_limit")),
            ("Initial Waiting",         "30 days"),
            ("PED Waiting",             f"{p.get('ped_waiting_months')} months" if p.get('ped_waiting_months') else None),
            ("Specific Illness Waiting",f"{p.get('specific_illness_waiting')} months" if p.get('specific_illness_waiting') else None),
            ("Co-payment",              f"{p.get('copayment_percent',0)}%"),
            ("Maternity",               "Covered" if p.get("maternity_covered") else "Not covered"),
            ("Restore",                 p.get("restore_type") or ("Yes" if p.get("restore_benefit") else "No")),
            ("NCB",                     f"{p.get('no_claim_bonus_percent')}%/yr" if p.get("no_claim_bonus_percent") else None),
            ("Network Hospitals",       f"{p.get('network_hospitals'):,}" if p.get("network_hospitals") else None),
            ("Claim Settlement Ratio",  f"{p.get('claim_settlement_ratio')}%" if p.get("claim_settlement_ratio") else None),
            ("Diabetes",                p.get("diabetes_notes") or ("Covered" if p.get("diabetes_covered") else None)),
            ("Cardiac",                 p.get("cardiac_notes") or ("Covered" if p.get("cardiac_covered") else None)),
        ]:
            if val and "None" not in str(val):
                lines.append(f"- **{label}**: {val}")
        lines.append("")
    return "\n".join(lines)

def rag_quality_ok(chunks: list) -> bool:
    """True if at least 2 chunks have similarity >= 0.5."""
    return sum(1 for c in chunks if c.get("similarity",0) >= 0.5) >= 2

# ── Context builders ───────────────────────────────────────────────────────────

def build_rag_context(question, policy_ids, section_tags, conditions=None):
    # Expand query with synonyms
    expanded = expand_query(question, conditions or [])

    # Retrieve
    result = retrieve(expanded, policy_ids=policy_ids,
                      section_tags=section_tags, top_k=10)
    chunks = result.get("chunks",[])

    # Rerank
    chunks = rerank_chunks(chunks, question)

    # Check quality — fallback to SQL context if RAG is weak
    rag_text = format_chunks({"chunks":chunks}, max_chunks=6)
    if not rag_quality_ok(chunks) and policy_ids:
        sql_text = sql_context_for_candidates(policy_ids)
        if sql_text:
            return sql_text + "\n\n## Additional Clause Details\n" + rag_text
    return rag_text

def build_fc_context(policy_ids, question, conditions=None):
    policies = get_policies(policy_ids)
    rag_text = build_rag_context(question, policy_ids, None, conditions)
    lines    = ["## Policy Structured Data\n"]
    for p in policies:
        lines.append(f"### {p.get('insurer')} — {p.get('product_slug')}")
        for label, val in [
            ("SI Range",                f"₹{p.get('si_min_lakhs')}L–₹{p.get('si_max_lakhs')}L"),
            ("Room Rent",               p.get("room_rent_limit")),
            ("Initial Waiting",         "30 days"),
            ("PED Waiting",             f"{p.get('ped_waiting_months')}m" if p.get('ped_waiting_months') else None),
            ("Specific Illness Waiting",f"{p.get('specific_illness_waiting')}m" if p.get('specific_illness_waiting') else None),
            ("Co-payment",              f"{p.get('copayment_percent',0)}%"),
            ("Maternity",               "Yes" if p.get("maternity_covered") else "No"),
            ("Restore",                 p.get("restore_type") or ("Yes" if p.get("restore_benefit") else "No")),
            ("NCB",                     f"{p.get('no_claim_bonus_percent')}%/yr" if p.get("no_claim_bonus_percent") else None),
            ("Network Hospitals",       p.get("network_hospitals")),
            ("CSR",                     f"{p.get('claim_settlement_ratio')}%" if p.get("claim_settlement_ratio") else None),
        ]:
            if val and "None" not in str(val): lines.append(f"- **{label}**: {val}")
        lines.append("")
    lines += ["\n## Relevant Policy Clauses\n", rag_text]
    return "\n".join(lines)

def format_sql_context(result):
    candidates = result.get("candidates",[])
    if not candidates: return "No matching policies found for the given criteria."
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

# ── Synthesizer prompt ─────────────────────────────────────────────────────────
# One universal prompt instead of 10 rigid ones.
# The LLM uses judgment based on the intent_description.

SYSTEM_PROMPT = """You are an expert Indian health insurance advisor with deep knowledge of IRDAI regulations and all major Indian health insurance products.

Answer the user's question based on the policy information provided in the context.
Apply sound judgment — you know Indian health insurance well.

UNIVERSAL RULES — apply to every response:

1. WAITING PERIODS — always distinguish clearly:
   - Initial waiting period: 30 days, applies to ALL policies, ALL conditions (except accidents)
   - Specified disease waiting period: typically 24 months, applies ONLY to listed conditions
     (gall bladder/bile duct stones, hernia, cataract, knee/joint replacement, tonsils,
     hysterectomy, benign tumors, varicose veins, piles/fissures, hydrocele, sinusitis,
     spinal disc disorders, ENT procedures, osteoarthritis)
   - PED waiting period: 24-48 months, for declared pre-existing conditions (diabetes, BP, cardiac)
   
   ACUTE ILLNESSES (jaundice, dengue, typhoid, malaria, fever, appendicitis, fracture,
   pneumonia, food poisoning, infections, stroke, emergency) have NO specified disease wait.
   They are covered after the standard 30-day initial waiting period under ALL policies.
   State this clearly and confidently — do not say "not found in retrieved data" for acute illnesses.

2. MISSING DATA — if a specific value isn't in the context:
   - For acute illnesses: confidently state "covered after 30-day initial wait"
   - For structured fields (premiums, network size): use the structured data provided
   - Only say "not found in retrieved policy data" for truly obscure clause details
   - Never repeat "not found" more than once in a response

3. RECOMMENDATIONS — always rank by fit. State the decisive factor clearly.
   Structure: top pick first, then alternatives, then what to watch out for.

4. COMPARISONS — use structured format. One section per parameter. End with a verdict table.

5. GENERAL QUESTIONS (what is NCB, how does waiting period work, what is TPA) —
   answer from your knowledge directly. No retrieval context needed.

6. CORPUS OVERVIEW — if asked about available policies, list from the provided corpus.
   Group by insurer. Don't attempt clause-level analysis.

7. ALWAYS cite specific numbers: months, ₹ amounts, percentages.
   Bold the key numbers. Use bullet points for lists.

8. HONESTY — if you genuinely don't know something, say so once and move on.
   Never pad with filler. Be direct and useful.

The user's intent: {intent_description}"""

# ── Pipeline ───────────────────────────────────────────────────────────────────

async def process_chat(req: ChatRequest):
    # Load session memory
    session = load_session(req.session_id)

    # Route
    router_result = route(
        req.message,
        summary=session["summary"],
        conversation_history=session["recent_messages"],
    )

    # Apply memory
    router_result            = resolve_followup(router_result, session)
    router_result["filters"] = enrich_filters(router_result.get("filters",{}), session)
    session                  = update_profile(session, router_result.get("filters",{}))

    decision     = router_result["retrieval_decision"]
    policies     = router_result["policies_mentioned"]
    filters      = router_result["filters"]
    section_tags = router_result["section_tags"]
    question     = router_result["specific_question"] or req.message
    conditions   = filters.get("conditions") or []
    intent_desc  = router_result.get("intent_description","")

    yield f"data: {json.dumps({'type':'route','decision':decision,'policies':policies,'intent':intent_desc[:80]})}\n\n"

    # Clarification
    if router_result.get("needs_clarification"):
        q = router_result.get("clarification_question","Could you provide more details?")
        session = add_turn(session, req.message, q)
        if should_summarize(session): session["summary"] = summarize(session)
        save_session(req.session_id, session)
        yield f"data: {json.dumps({'type':'answer','text':q})}\n\n"
        yield "data: [DONE]\n\n"; return

    # Missing from corpus
    if router_result.get("_missing_from_corpus"):
        names = router_result.get("not_in_corpus",[])
        msg   = (f"I don't have {', '.join(names)} in my database. "
                 f"You can upload its policy document and I'll analyse it. "
                 f"Currently I cover: {', '.join(set(p['insurer'] for p in CORPUS))}.")
        session = add_turn(session, req.message, msg)
        save_session(req.session_id, session)
        yield f"data: {json.dumps({'type':'answer','text':msg})}\n\n"
        yield "data: [DONE]\n\n"; return

    # Build context based on retrieval decision
    context = ""

    if decision == "NO_RETRIEVAL":
        # Give corpus list so LLM can answer "what policies do you have" questions
        context = f"Available policies in database:\n{CORPUS_LIST}"

    elif decision == "SQL_ONLY":
        sql_res = sql_filter(filters, "sql", limit=10)
        context = format_sql_context(sql_res)
        session["last_candidate_ids"] = [c["id"] for c in sql_res.get("candidates",[])]

    elif decision == "FC":
        context = build_fc_context(policies, question, conditions)

    else:  # RAG
        # SQL filter first if we have meaningful filters
        candidate_ids = None
        has_filters   = any(v for v in filters.values() if v is not None)

        if has_filters and not policies:
            sql_res       = sql_filter(filters, "rag", limit=12)
            candidate_ids = [c["id"] for c in sql_res.get("candidates",[])]
            session["last_candidate_ids"] = candidate_ids
            yield f"data: {json.dumps({'type':'trace','text':f'Filtered to {len(candidate_ids)} candidates'})}\n\n"

        search_ids = policies if policies else candidate_ids
        context    = build_rag_context(question, search_ids, section_tags or None, conditions)

    yield f"data: {json.dumps({'type':'trace','text':'Synthesizing...'})}\n\n"

    # Build messages
    system_content = SYSTEM_PROMPT.format(intent_description=intent_desc or req.message)
    user_content   = f"Question: {req.message}\n"
    profile        = {k:v for k,v in filters.items() if v is not None}
    if profile:                    user_content += f"User profile: {json.dumps(profile)}\n"
    if session.get("summary"):     user_content += f"Conversation so far: {session['summary']}\n"
    if context:                    user_content += f"\nPolicy Information:\n{context}"

    messages = [SystemMessage(content=system_content), HumanMessage(content=user_content)]

    # Stream with key rotation
    full_answer = ""
    last_err    = None
    for key, model_name in GEMINI_CANDIDATES:
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=key,
                                          temperature=0.3, streaming=True)
            async for chunk in llm.astream(messages):
                if chunk.content:
                    full_answer += chunk.content
                    yield f"data: {json.dumps({'type':'answer','text':chunk.content})}\n\n"
            break
        except Exception as e:
            if any(x in str(e) for x in ["429","quota","503","504","Deadline"]):
                last_err = e; continue
            raise

    if not full_answer and last_err:
        yield f"data: {json.dumps({'type':'error','text':f'All models failed: {last_err}'})}\n\n"

    # Save memory
    session = add_turn(session, req.message, full_answer)
    if should_summarize(session):
        yield f"data: {json.dumps({'type':'trace','text':'Updating memory...'})}\n\n"
        session["summary"] = summarize(session)
    save_session(req.session_id, session)

    yield "data: [DONE]\n\n"

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(process_chat(req), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/health")
def health(): return {"status":"ok","policies":len(CORPUS)}

@app.get("/policies")
def list_policies():
    return [{"id":p["id"],"insurer":p["insurer"],"slug":p["product_slug"]} for p in CORPUS]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
