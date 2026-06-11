"""FastAPI Backend — Insurance Intel (Agentic RAG with Deterministic Guardrails)"""
import os, json, asyncio, re
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

# ── LAYER 3: THE REASONING ENGINE (DETERMINISTIC LOGIC) ──────────────────────

def detect_feature_gaps(top_policies, filters):
    warnings = []
    age = filters.get("age", 30)
    zone = str(filters.get("zone", "")).lower()
    
    for p in top_policies:
        name = p.get("product_slug", p.get("id"))
        copay = p.get("copayment_percent", 0)
        if age >= 60 and copay > 0:
            warnings.append(f"⚠️ {name} has a mandatory {copay}% co-pay for seniors.")
            
        rr_limit = str(p.get("room_rent_limit", "")).lower()
        is_metro = any(z in zone for z in ["zone 1", "zone a", "metro", "tier 1", "delhi", "mumbai", "bangalore"])
        if is_metro and rr_limit and "no limit" not in rr_limit and "single" not in rr_limit:
            warnings.append(f"⚠️ {name} caps room rent at '{p.get('room_rent_limit')}'.")
            
    return warnings

def build_dynamic_instructions(filters, warnings):
    instructions = []
    requested_si = filters.get("sum_insured_inr")
    if requested_si:
        instructions.append(f"MUST verify if quoted SI matches requested ₹{requested_si/100000:g}L.")
        
    if filters.get("conditions"):
        instructions.append("MUST present 'Fastest Coverage' (short PED wait) vs 'Budget' track with warnings.")
        
    if warnings:
        instructions.append("MUST surface Feature Gap Warnings.")
        
    return instructions

def sql_context_for_candidates(candidate_ids: list) -> str:
    if not candidate_ids: return ""
    policies = get_policies(candidate_ids[:8])
    if not policies: return ""
    lines = ["## Detailed Benchmark Data (for comparison)\n"]
    for p in policies:
        lines.append(f"### {p.get('insurer')} — {p.get('product_slug')}")
        for label, val in [
            ("Room Rent",               p.get("room_rent_limit")),
            ("PED Waiting",             f"{p.get('ped_waiting_months')} months" if p.get('ped_waiting_months') is not None else "Not specified"),
            ("Specific Illness Waiting",f"{p.get('specific_illness_waiting')} months" if p.get('specific_illness_waiting') is not None else "Not specified"),
            ("Co-payment",              f"{p.get('copayment_percent',0)}%"),
            ("Restore Benefit",         p.get("restore_type") or ("Yes" if p.get("restore_benefit") else "No")),
            ("Maternity",               "Covered" if p.get("maternity_covered") else "Not covered"),
        ]:
            if val is not None: lines.append(f"- {label}: {val}")
        lines.append("")
    return "\n".join(lines)

def expand_query(question: str, conditions: list) -> str:
    if not conditions: return question
    prompt = (f"For this Indian health insurance query, add 3-5 relevant medical/insurance synonyms. "
              f"Output ONLY the expanded query. Original: {question} Conditions: {', '.join(conditions)}")
    for key, model in GEMINI_CANDIDATES:  # iterate best→worst, not [-1]
        try:
            llm  = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0, max_output_tokens=150)
            resp = llm.invoke([HumanMessage(content=prompt)])
            return resp.content.strip() or question
        except Exception:
            continue
    return question

def rerank_chunks(chunks: list, question: str) -> list:
    if not chunks: return chunks
    filtered = [c for c in chunks if c.get("similarity",0) >= 0.42]
    if len(filtered) < 3: filtered = sorted(chunks, key=lambda c: c.get("similarity",0), reverse=True)[:3]
    seen, deduped = {}, []
    for c in filtered:
        key = f"{c.get('policy_id')}_{c.get('section_tag')}"
        if seen.get(key) is None or abs(c.get("similarity",0) - seen.get(key)) > 0.05:
            deduped.append(c)
            seen[key] = c.get("similarity",0)
    return deduped

def build_rag_context(question, policy_ids, section_tags, conditions=None):
    expanded = expand_query(question, conditions or [])
    result = retrieve(expanded, policy_ids=policy_ids, section_tags=section_tags, top_k=10)
    chunks = rerank_chunks(result.get("chunks",[]), question)
    return format_chunks({"chunks":chunks}, max_chunks=6)

def build_fc_context(policy_ids, question, conditions=None):
    rag_text = build_rag_context(question, policy_ids, None, conditions)
    return sql_context_for_candidates(policy_ids) + "\n## Relevant Policy Clauses\n" + rag_text

def format_sql_context(result):
    candidates = result.get("candidates",[])
    if not candidates: return "No matching policies found."
    return f"System found {len(candidates)} matching policies in the catalog."

SYSTEM_PROMPT = """You are "Medi-Advisor", an expert health insurance consultant for the Indian market. You work for a neutral comparison platform and have access to exact premium data and policy documents.

Your role is that of a trusted human advisor. Do not act like a search engine. Interpret the data provided in the Dynamic Context Packet below and give nuanced, proactive recommendations.

## STRICT FORMATTING RULES (CRITICAL)
1. NO RAW TABLES: The frontend DOES NOT support Markdown tables. NEVER output a table format (`|---|---|`).
2. BE PUNCHY & VISUAL: Avoid massive walls of text. Use bullet points and emojis.
3. EXPLICIT PRICING: You MUST weave the exact calculated Rupee premium amounts into your text next to the policy names.

**You MUST structure your response EXACTLY like this:**

**🏆 Top Recommendation & Strategy**
[Declare the winner. If instructed to present Two Tracks (Fastest Coverage vs Budget), explain the tradeoff clearly here.]

**⚖️ Feature Breakdown (Policy by Policy)**
* 🏥 **[Policy 1 Name]** (₹[Premium]/year): [Detail its Room Rent, PED Wait, Specific Illness Wait, Restoration, and Co-pay. Include any warnings from the Feature Gaps!]
* 🏥 **[Policy 2 Name]** (₹[Premium]/year): [Detail features & warnings]
*(Ensure every policy gets its own dedicated bullet point).*

**💡 Advisor's Verdict**
[A final, warm closing thought guiding their decision.]"""


# ── ORCHESTRATOR ────────────────────────────────────────────────────────────

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

    # Wire insurer_mentioned from router into filters so sql_filter can use p_insurer
    if router_result.get("insurer_mentioned") and not filters.get("insurer"):
        filters["insurer"] = router_result["insurer_mentioned"]

    context_blocks = []
    pricing_queue  = set(policies or router_result.get("resolved_policy_ids", []))

    if decision == "NO_RETRIEVAL":
        context_blocks.append(f"Available policies:\n{CORPUS_LIST}")
    elif decision == "SQL_ONLY" or not pricing_queue:
        # Run SQL to populate pricing_queue — the llm_packet built after pricing
        # is what carries real data to the LLM; we do NOT need a placeholder here.
        sql_res = sql_filter(filters, "sql", limit=100)
        for cand in sql_res.get("candidates", []): pricing_queue.add(cand["id"])
    elif decision == "FC":
        context_blocks.append(build_fc_context(list(pricing_queue), question, conditions))
    else:
        has_filters = any(v for k, v in filters.items() if v is not None and k not in ["age", "sum_insured_inr", "zone"])
        search_ids = list(pricing_queue)
        if has_filters and not search_ids:
            sql_res = sql_filter(filters, "rag", limit=100)
            search_ids = [c["id"] for c in sql_res.get("candidates",[])]
            for sid in search_ids: pricing_queue.add(sid)
        
        rag_text = build_rag_context(question, search_ids[:12], section_tags or None, conditions)
        context_blocks.append(rag_text)

    user_msg_lower = req.message.lower()
    if "top up" not in user_msg_lower and "top-up" not in user_msg_lower and "extra care" not in user_msg_lower:
        pricing_queue = {pid for pid in pricing_queue if "extra_care" not in pid.lower()}

    if filters.get("age") and pricing_queue:
        yield f"data: {json.dumps({'type':'trace','text':f'Calculating quotes for {len(pricing_queue)} policies...'})}\n\n"
        
        target_si = filters.get("sum_insured_inr") or 500000
        target_zone = filters.get("zone") or "All"
        calculated_quotes = []
        
        def fetch_quote(pid):
            try:
                res = calculate_quote(pid, int(filters["age"]), requested_si=target_si, zone=target_zone)
                if res.get("status") == "success":
                    res["policy_id"] = pid
                    return res
            except Exception: pass
            return None

        # Threadpool executed safely with max workers to avoid DB/event loop hangs
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fetch_quote, pid) for pid in list(pricing_queue)]
            for future in concurrent.futures.as_completed(futures, timeout=15):
                try:
                    res = future.result(timeout=3)
                    if res: calculated_quotes.append(res)
                except:
                    continue
                
        if calculated_quotes:
            calculated_quotes.sort(key=lambda x: x["final_payable"])
            
            ped_keywords = ["diabetes", "bp", "blood pressure", "asthma", "waiting", "ped", "pre-existing", "disease", "day 0", "day 1"]
            needs_ped_sort = bool(conditions or any(k in user_msg_lower for k in ped_keywords))
            
            if needs_ped_sort:
                all_ids = [q["policy_id"] for q in calculated_quotes]
                full_policies = get_policies(all_ids)
                
                def get_ped_val(val):
                    if val is None: return 99
                    val_str = str(val).lower().strip()
                    if "day 1" in val_str or "day 0" in val_str: return 0
                    nums = re.findall(r'\d+', val_str)
                    if nums: return int(nums[0])
                    return 99

                ped_lookup = {p["id"]: get_ped_val(p.get("ped_waiting_months")) for p in full_policies}
                
                top_quotes = calculated_quotes[:3]
                remaining  = calculated_quotes[3:]  # index-based, never uses identity
                remaining.sort(key=lambda x: (ped_lookup.get(x["policy_id"], 99), x["final_payable"]))
                
                top_quotes.extend(remaining[:2]) 
                top_quotes.sort(key=lambda x: x["final_payable"])
            else:
                top_quotes = calculated_quotes[:5]
            
            yield f"data: {json.dumps({'type':'structured_quote', 'data': top_quotes})}\n\n"
            
            top_ids = [q["policy_id"] for q in top_quotes]
            session["last_candidate_ids"] = top_ids  # FIX: persist so follow-up queries resolve
            full_winning_policies = get_policies(top_ids)
            
            feature_warnings = detect_feature_gaps(full_winning_policies, filters)
            dynamic_instructions = build_dynamic_instructions(filters, feature_warnings)
            rich_benchmarks = sql_context_for_candidates(top_ids)
            quote_text = "\n".join([f"- **{q['policy_id'].replace('_', ' ')}**: ₹{q['final_payable']:,.0f}/year" for q in top_quotes])
            
            llm_packet = (
                "=========================================\n"
                "🛡️ DYNAMIC CONTEXT PACKET FOR MEDI-ADVISOR\n"
                "=========================================\n\n"
                "## 1. CALCULATED PREMIUMS\n"
                f"{quote_text}\n\n"
                "## 2. BENCHMARK DATA (Room Rent / PED Wait / Co-pay per policy)\n"
                f"{rich_benchmarks}\n"
                "## 3. FEATURE GAP WARNINGS (SURFACE THESE — mandatory!)\n"
                f"{chr(10).join(['- ' + w for w in feature_warnings]) if feature_warnings else 'No critical feature gaps detected.'}\n\n"
                "## 4. STRICT INSTRUCTIONS FOR THIS QUERY\n"
                f"{chr(10).join(['- ' + i for i in dynamic_instructions]) if dynamic_instructions else 'Proceed with standard comparison.'}\n"
                "========================================="
            )
            context_blocks.insert(0, llm_packet)
        else:
            context_blocks.insert(0, f"SYSTEM ERROR: No database rows matched for Age {filters['age']}.")

    yield f"data: {json.dumps({'type':'trace','text':'Synthesizing...'})}\n\n"

    system_content = SYSTEM_PROMPT + f"\n\nUSER'S SPECIFIC INTENT/QUERY: {intent_desc or req.message}"
    user_content   = f"Question: {req.message}\n"
    profile        = {k:v for k,v in filters.items() if v is not None}
    if profile: user_content += f"User profile matrix: {json.dumps(profile)}\n"
    
    final_context = "\n\n".join(context_blocks)
    if final_context: user_content += f"\nAssembled Context Layer:\n{final_context}"

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
    if should_summarize(session): session["summary"] = summarize(session)
    save_session(req.session_id, session)

    yield "data: [DONE]\n\n"

@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(process_chat(req), media_type="text/event-stream", headers={"Cache-Control":"no-cache"})

@app.get("/health")
def health(): return {"status":"ok","policies":len(CORPUS)}

@app.get("/policies")
def list_policies(): return [{"id":p["id"],"insurer":p["insurer"],"slug":p["product_slug"]} for p in CORPUS]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)