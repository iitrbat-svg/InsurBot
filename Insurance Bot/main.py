"""FastAPI Backend — Insurance Intel (with Pricing Engine, SQL fallback, and Error Trapping)"""
import os, json, asyncio
import concurrent.futures
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2","true")
os.environ.setdefault("LANGCHAIN_PROJECT","insurance-intel")

from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from router        import route, CORPUS, CORPUS_LIST
from sql_filter    import sql_filter, get_policies
from rag_retriever import retrieve, format_chunks
from memory        import (load_session, save_session, add_turn, update_profile,
                           enrich_filters, resolve_followup, should_summarize, summarize)
from pricing       import calculate_quote

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
    print("Ready. Models will load lazily on the first request.")

class ChatRequest(BaseModel):
    message:    str
    session_id: str = "default"

def expand_query(question: str, conditions: list) -> str:
    if not conditions: return question
    key, model = GEMINI_CANDIDATES[-1]
    try:
        llm  = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0, max_output_tokens=150)
        resp = llm.invoke([HumanMessage(content=
            f"""For this Indian health insurance query, add 3-5 relevant medical/insurance synonyms
and alternate terms that would appear in policy documents. Keep it concise.
Output ONLY the expanded query, no explanation.

Original: {question}
Conditions mentioned: {', '.join(conditions)}

Expanded query:""")])
        expanded = resp.content.strip()
        return expanded if expanded else question
    except Exception:
        return question

def rerank_chunks(chunks: list, question: str) -> list:
    if not chunks: return chunks
    filtered = [c for c in chunks if c.get("similarity",0) >= 0.42]
    if len(filtered) < 3:
        filtered = sorted(chunks, key=lambda c: c.get("similarity",0), reverse=True)[:3]
    seen, deduped = {}, []
    for c in filtered:
        key = f"{c.get('policy_id')}_{c.get('section_tag')}"
        prev_sim = seen.get(key)
        if prev_sim is None or abs(c.get("similarity",0) - prev_sim) > 0.05:
            deduped.append(c)
            seen[key] = c.get("similarity",0)
    return deduped

def sql_context_for_candidates(candidate_ids: list) -> str:
    if not candidate_ids: return ""
    policies = get_policies(candidate_ids[:8])
    if not policies: return ""
    lines = ["## Detailed Benchmark Data (for comparison)\n"]
    for p in policies:
        lines.append(f"### {p.get('insurer')} — {p.get('product_slug')}")
        for label, val in [
            ("Room Rent",               p.get("room_rent_limit")),
            ("PED Waiting",             f"{p.get('ped_waiting_months')} months" if p.get('ped_waiting_months') else "Not specified"),
            ("Specific Illness Waiting",f"{p.get('specific_illness_waiting')} months" if p.get('specific_illness_waiting') else "Not specified"),
            ("Co-payment",              f"{p.get('copayment_percent',0)}%"),
            ("Restore Benefit",         p.get("restore_type") or ("Yes" if p.get("restore_benefit") else "No")),
            ("Maternity",               "Covered" if p.get("maternity_covered") else "Not covered"),
        ]:
            if val is not None: lines.append(f"- {label}: {val}")
        lines.append("")
    return "\n".join(lines)

def rag_quality_ok(chunks: list) -> bool:
    return sum(1 for c in chunks if c.get("similarity",0) >= 0.5) >= 2

def build_rag_context(question, policy_ids, section_tags, conditions=None):
    expanded = expand_query(question, conditions or [])
    result = retrieve(expanded, policy_ids=policy_ids, section_tags=section_tags, top_k=10)
    chunks = rerank_chunks(result.get("chunks",[]), question)
    rag_text = format_chunks({"chunks":chunks}, max_chunks=6)
    if not rag_quality_ok(chunks) and policy_ids:
        sql_text = sql_context_for_candidates(policy_ids)
        if sql_text: return sql_text + "\n\n## Additional Clause Details\n" + rag_text
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
        ]:
            if val and "None" not in str(val): lines.append(f"- **{label}**: {val}")
        lines.append("")
    lines += ["\n## Relevant Policy Clauses\n", rag_text]
    return "\n".join(lines)

# --- REWRITTEN TO AVOID SPAMMING THE LLM WITH 30 INCOMPLETE POLICIES ---
def format_sql_context(result):
    candidates = result.get("candidates",[])
    if not candidates: return "No matching policies found for the given criteria."
    return f"System found {len(candidates)} matching policies in the catalog. Passing to the pricing engine."

# --- NEW PROMPT: FORCING A POLICY-BY-POLICY LIST ---
SYSTEM_PROMPT = """You are an expert Indian health insurance advisor.

Answer accurately based on the factual calculations and detailed benchmark data provided in the context below.

UNIVERSAL RULES (CRITICAL):
1. NO RAW TABLES: The frontend DOES NOT support Markdown tables. NEVER output a table format (`|---|---|`).
2. EXPLICIT POLICY LISTING: You MUST dedicate a separate bullet point to EACH policy provided in the context. If there are 5 policies, you must list 5 bullet points. Do not group by features.
3. EXPLICIT PRICING IN TEXT: You MUST include the exact Rupee premium amounts next to each policy name.
4. COMPREHENSIVE FEATURES: For each policy, you must mention its specific Room Rent, PED Waiting Period, Restoration benefit, and Co-pay.

**You MUST structure your response EXACTLY like this:**

**🏆 Top Recommendation**
[1-2 sentences picking the absolute best policy based on the balance of price, waiting periods, and room rent.]

**⚖️ Feature Breakdown (Policy by Policy)**
* 🏥 **[Policy 1 Name]** (₹[Premium Amount]/year): [2-3 sentences detailing its specific Room Rent, PED Wait, Specific Illness Wait, Restoration, and Co-pay]
* 🏥 **[Policy 2 Name]** (₹[Premium Amount]/year): [2-3 sentences detailing its specific Room Rent, PED Wait, Specific Illness Wait, Restoration, and Co-pay]
* 🏥 **[Policy 3 Name]** (₹[Premium Amount]/year): [2-3 sentences detailing its specific Room Rent, PED Wait, Specific Illness Wait, Restoration, and Co-pay]
* 🏥 **[Policy 4 Name]** (₹[Premium Amount]/year): [2-3 sentences detailing its specific Room Rent, PED Wait, Specific Illness Wait, Restoration, and Co-pay]
* 🏥 **[Policy 5 Name]** (₹[Premium Amount]/year): [2-3 sentences detailing its specific Room Rent, PED Wait, Specific Illness Wait, Restoration, and Co-pay]

**💡 Final Verdict**
[A quick closing thought on how the user should decide based on their budget vs. features.]"""

# ── Upgraded Pipeline Orchestration ───────────────────────────────────────────

async def process_chat(req: ChatRequest):
    session = load_session(req.session_id)

    router_result = route(req.message, summary=session["summary"], conversation_history=session["recent_messages"])
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

    if router_result.get("needs_clarification"):
        q = router_result.get("clarification_question","Could you provide more details?")
        session = add_turn(session, req.message, q)
        if should_summarize(session): session["summary"] = summarize(session)
        save_session(req.session_id, session)
        yield f"data: {json.dumps({'type':'answer','text':q})}\n\n"
        yield "data: [DONE]\n\n"; return

    if router_result.get("_missing_from_corpus"):
        names = router_result.get("not_in_corpus",[])
        msg   = f"I don't have {', '.join(names)} in my database. Currently I cover: {', '.join(set(p['insurer'] for p in CORPUS))}."
        session = add_turn(session, req.message, msg)
        save_session(req.session_id, session)
        yield f"data: {json.dumps({'type':'answer','text':msg})}\n\n"
        yield "data: [DONE]\n\n"; return

    context_blocks = []
    pricing_queue  = set(policies or router_result.get("resolved_policy_ids", []))

    # Phase A: Handle Structural Filtering & Catalog Discretion
    if decision == "NO_RETRIEVAL":
        context_blocks.append(f"Available policies in database:\n{CORPUS_LIST}")
    elif decision == "SQL_ONLY" or not pricing_queue:
        sql_res = sql_filter(filters, "sql", limit=30)
        context_blocks.append(format_sql_context(sql_res))
        for cand in sql_res.get("candidates", []):
            pricing_queue.add(cand["id"])
        session["last_candidate_ids"] = [c["id"] for c in sql_res.get("candidates",[])]
    elif decision == "FC":
        context_blocks.append(build_fc_context(list(pricing_queue), question, conditions))
    else:
        has_filters = any(v for k, v in filters.items() if v is not None and k not in ["age", "sum_insured_inr", "zone"])
        search_ids = list(pricing_queue)
        if has_filters and not search_ids:
            sql_res = sql_filter(filters, "rag", limit=30)
            search_ids = [c["id"] for c in sql_res.get("candidates",[])]
            session["last_candidate_ids"] = search_ids
            for sid in search_ids: pricing_queue.add(sid)
        
        # Only fetch actual PDF text for top 5 to save tokens
        rag_text = build_rag_context(question, search_ids[:5], section_tags or None, conditions)
        context_blocks.append(rag_text)

    # Phase B: Intercept with the Premium Underwriting Engine
    if filters.get("age") and pricing_queue:
        yield f"data: {json.dumps({'type':'trace','text':f'Calculating quotes in parallel for {len(pricing_queue)} policies...'})}\n\n"
        target_si = filters.get("sum_insured_inr") or 500000
        target_zone = filters.get("zone") or "All"
        
        calculated_quotes = []
        
        def fetch_quote(pid):
            try:
                res = calculate_quote(pid, int(filters["age"]), requested_si=target_si, zone=target_zone)
                if res.get("status") == "success":
                    res["policy_id"] = pid
                    return res
            except Exception:
                pass
            return None

        # PARALLEL EXECUTION
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fetch_quote, pid) for pid in list(pricing_queue)]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    calculated_quotes.append(res)
                
        if calculated_quotes:
            # Sort all returned quotes by cheapest price
            calculated_quotes.sort(key=lambda x: x["final_payable"])
            top_quotes = calculated_quotes[:5]
            
            # YIELD TO THE FRONTEND TO BUILD THE CARDS!
            yield f"data: {json.dumps({'type':'structured_quote', 'data': top_quotes})}\n\n"
            
            # --- NEW SMART FETCH: GET RICH BENCHMARKS ONLY FOR THE 5 WINNERS ---
            top_ids = [q["policy_id"] for q in top_quotes]
            rich_benchmarks = sql_context_for_candidates(top_ids)
            
            quote_text = "\n".join([f"- **{q['policy_id'].replace('_', ' ')}**: ₹{q['final_payable']:,.0f}/year" for q in top_quotes])
            
            llm_note = (
                f"## FINAL CALCULATED QUOTES (Top {len(top_quotes)} Cheapest)\n"
                f"{quote_text}\n\n"
                f"## DETAILED POLICY BENCHMARKS\n"
                f"{rich_benchmarks}\n\n"
                "SYSTEM NOTE: You MUST list the top policies and their premium amounts in your text response. "
                "Use the 'Detailed Policy Benchmarks' above to compare their waiting periods (PED, Specific Illness), Restore benefits, and Room Rent limits. Do not just rely on room rent!"
            )
            context_blocks.insert(0, llm_note)
            
        else:
            context_blocks.insert(0, (
                "## SYSTEM PRICING ERROR\n"
                "You tried to calculate premiums, but the database returned no matching rows "
                f"for Age {filters['age']}. "
                "You MUST apologize to the user and state that you do not have the exact premium "
                "data for these policies in your database right now, but provide the feature analysis below."
            ))

    # Phase C: Final Synthesis Compilation
    yield f"data: {json.dumps({'type':'trace','text':'Synthesizing...'})}\n\n"

    system_content = SYSTEM_PROMPT.format(intent_description=intent_desc or req.message)
    user_content   = f"Question: {req.message}\n"
    profile        = {k:v for k,v in filters.items() if v is not None}
    if profile:                user_content += f"User profile matrix: {json.dumps(profile)}\n"
    if session.get("summary"):     user_content += f"Conversation historical state: {session['summary']}\n"
    
    final_context = "\n\n".join(context_blocks)
    if final_context:          user_content += f"\nAssembled Context Layer:\n{final_context}"

    messages = [SystemMessage(content=system_content), HumanMessage(content=user_content)]

    full_answer = ""
    last_err    = None
    for key, model_name in GEMINI_CANDIDATES:
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=key, temperature=0.2, streaming=True)
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

    session = add_turn(session, req.message, full_answer)
    if should_summarize(session):
        yield f"data: {json.dumps({'type':'trace','text':'Updating memory...'})}\n\n"
        session["summary"] = summarize(session)
    save_session(req.session_id, session)

    yield "data: [DONE]\n\n"

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