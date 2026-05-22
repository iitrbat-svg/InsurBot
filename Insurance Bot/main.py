"""
STEP 7+: FastAPI App — Insurance Intel Backend
================================================
Wires: Router → SQL Filter → RAG/FC Retriever → Synthesizer
Streaming responses. Model loaded once at startup.

Install:
  pip install fastapi uvicorn sse-starlette langchain-google-genai
              sentence-transformers torch python-dotenv requests

Run:
  uvicorn main:app --reload --port 8000

Endpoints:
  POST /chat          — main chat endpoint (streaming SSE)
  GET  /health        — health check
  GET  /policies      — list all policies in corpus
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

# Local modules
from router       import route, CORPUS
from sql_filter   import sql_filter, get_policies
from rag_retriever import retrieve, format_chunks, warmup

app = FastAPI(title="Insurance Intel API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

GEMINI_CANDIDATES = [(k, m) for k, m in [
    (os.getenv("GEMINI_KEY_PAID"), "gemini-2.5-flash"),
    (os.getenv("GEMINI_KEY_PAID"), "gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_1"),    "gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_1"),    "gemini-2.0-flash"),
    (os.getenv("GEMINI_KEY_2"),    "gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_2"),    "gemini-2.0-flash"),
    (os.getenv("GEMINI_KEY_3"),    "gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_3"),    "gemini-2.0-flash"),
] if k]

# ── Startup: warm up embedder so first query is fast ──────────────────────────

@app.on_event("startup")
async def startup():
    print("Warming up embedder...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, warmup)
    print("Ready.")

# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:  str
    session_id: str = "default"
    summary:  str = ""
    history:  list = []   # last 2-3 messages [{role, content}]

# ── Gemini LLM with rotation ───────────────────────────────────────────────────

def get_llm(idx: int = 0):
    key, model = GEMINI_CANDIDATES[idx % len(GEMINI_CANDIDATES)]
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=0.3,
        streaming=True,
    )

# ── Context builders ───────────────────────────────────────────────────────────

def _sb_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }

def get_gemini_uris(policy_ids: list[str]) -> dict[str, list]:
    """Fetch Gemini File URIs for FC retrieval."""
    if not policy_ids: return {}
    r = _req.get(
        f"{SUPABASE_URL}/rest/v1/policies"
        f"?id=in.({','.join(policy_ids)})"
        f"&select=id,gemini_file_uris",
        headers=_sb_headers(), timeout=10)
    if not r.ok: return {}
    return {row["id"]: row.get("gemini_file_uris") or [] for row in r.json()}

def build_fc_context(policy_ids: list[str], question: str) -> str:
    """
    Full-context retrieval: fetch structured data from policies table
    and format as rich context. Gemini file URIs only work server-side
    with the Gemini Files API — for now we use structured DB data +
    top RAG chunks as the context instead.
    This avoids re-uploading PDFs while still giving accurate answers.
    """
    policies = get_policies(policy_ids)
    if not policies:
        return "Policy data not found."

    rag = retrieve(question, policy_ids=policy_ids, top_k=10, min_similarity=0.2)
    rag_context = format_chunks(rag, max_chunks=8)

    lines = ["## Policy Details\n"]
    for p in policies:
        lines.append(f"### {p.get('insurer')} — {p.get('product_slug')}")
        fields = {
            "Sum Insured":        f"₹{p.get('si_min_lakhs')}L – ₹{p.get('si_max_lakhs')}L",
            "Room Rent":          p.get("room_rent_limit"),
            "PED Waiting":        f"{p.get('ped_waiting_months')} months",
            "Co-payment":         f"{p.get('copayment_percent', 0)}%",
            "Maternity":          "Yes" if p.get("maternity_covered") else "No",
            "Restore":            p.get("restore_type") or ("Yes" if p.get("restore_benefit") else "No"),
            "NCB":                f"{p.get('no_claim_bonus_percent')}% per year" if p.get("no_claim_bonus_percent") else None,
            "Network Hospitals":  p.get("network_hospitals"),
            "Claim Settlement":   f"{p.get('claim_settlement_ratio')}%" if p.get("claim_settlement_ratio") else None,
        }
        for k, v in fields.items():
            if v and v != "None" and "None" not in str(v):
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("\n## Relevant Policy Clauses\n")
    lines.append(rag_context)
    return "\n".join(lines)

def build_rag_context(
    question: str,
    policy_ids: list[str] | None,
    section_tags: list[str] | None,
) -> str:
    rag = retrieve(question, policy_ids=policy_ids,
                   section_tags=section_tags, top_k=8)
    return format_chunks(rag, max_chunks=6)

# ── Synthesizer prompts ────────────────────────────────────────────────────────

def build_system_prompt(intent: str, router_result: dict) -> str:
    base = """You are an expert Indian health insurance advisor.
Answer clearly and accurately based ONLY on the policy information provided.
Always cite specific numbers: waiting periods, limits, percentages.
If information is not in the context, say so explicitly — never guess.
Use clean formatting: bullet points for lists, bold for key numbers.
Be concise but complete."""

    intent_instructions = {
        "recommendation": """
Rank the policies from best to worst fit for the user's profile.
For each policy explain: (1) why it fits or doesn't, (2) key benefit,
(3) main concern. End with a clear top recommendation with reasoning.""",

        "comparison": """
Compare policies side by side on the specific parameters asked.
Use a structured format: one section per parameter being compared.
End with a summary of which policy wins on each parameter.""",

        "evaluation": """
Evaluate the policy thoroughly across all dimensions:
waiting periods, exclusions, sub-limits, room rent, co-payment,
restore benefit, NCB. If user shared a profile, assess fit explicitly.
End with a verdict: recommended / conditionally recommended / not recommended,
with clear reasoning.""",

        "corpus_search": """
List all matching policies clearly.
For each, state the specific clause or term that answers the question.
Don't pad with unrelated information.""",

        "condition_specific": """
For each policy, state exactly: (1) is the condition covered,
(2) waiting period, (3) any sub-limits or exclusions, (4) loading if any.
Be precise with months and amounts.""",

        "claims_process": """
Explain the claims process step by step.
Distinguish between cashless and reimbursement where relevant.""",

        "general_knowledge": """
Explain the concept clearly with a practical Indian insurance example.
Keep it brief.""",
    }

    instruction = intent_instructions.get(intent, "")
    return base + instruction

# ── Main chat handler ──────────────────────────────────────────────────────────

async def process_chat(req: ChatRequest):
    """
    Full pipeline:
    1. Route → classify intent
    2. Retrieve context (SQL / RAG / FC based on intent)
    3. Synthesize → stream response
    """
    # ── Step 1: Route ──────────────────────────────────────────────────────────
    router_result = route(
        req.message,
        summary=req.summary,
        conversation_history=req.history,
    )

    intent         = router_result["intent"]
    retrieval_mode = router_result["retrieval_mode"]
    policies       = router_result["policies_mentioned"]
    filters        = router_result["filters"]
    section_tags   = router_result["section_tags"]
    question       = router_result["specific_question"] or req.message

    # Yield router decision for frontend trace
    yield f"data: {json.dumps({'type': 'route', 'intent': intent, 'retrieval_mode': retrieval_mode, 'policies': policies})}\n\n"

    # ── Step 2: Handle clarification / missing docs ────────────────────────────
    if router_result.get("needs_clarification"):
        q = router_result.get("clarification_question", "Could you provide more details?")
        yield f"data: {json.dumps({'type': 'answer', 'text': q})}\n\n"
        yield "data: [DONE]\n\n"
        return

    if intent == "missing_docs":
        names = router_result.get("not_in_corpus", [])
        msg   = (f"I don't have {', '.join(names)} in my database. "
                 f"You can upload its policy document and I'll analyse it.")
        yield f"data: {json.dumps({'type': 'answer', 'text': msg})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # ── Step 3: Retrieve context ───────────────────────────────────────────────
    context = ""

    if retrieval_mode == "llm_only":
        context = ""   # No retrieval needed

    elif retrieval_mode == "sql_only":
        result  = sql_filter(filters, intent)
        context = _format_sql_result(result, intent)

    elif retrieval_mode == "full_context":
        if not policies:
            # No specific policy named — fall back to RAG
            context = build_rag_context(question, None, section_tags or None)
        else:
            context = build_fc_context(policies, question)

    elif retrieval_mode in ("rag_filtered", "ask_user"):
        # SQL filter first if recommendation/condition_specific
        candidate_ids = None
        if intent in ("recommendation", "condition_specific", "corpus_search"):
            sql_result    = sql_filter(filters, intent, limit=12)
            candidate_ids = [c["id"] for c in sql_result["candidates"]]
            yield f"data: {json.dumps({'type': 'trace', 'text': f'Filtered to {len(candidate_ids)} candidates'})}\n\n"

        # Use named policies if provided
        search_ids = policies if policies else candidate_ids
        context    = build_rag_context(question, search_ids, section_tags or None)

    yield f"data: {json.dumps({'type': 'trace', 'text': 'Synthesizing answer...'})}\n\n"

    # ── Step 4: Synthesize ─────────────────────────────────────────────────────
    system_prompt = build_system_prompt(intent, router_result)

    user_content = f"""Question: {req.message}

User profile: {json.dumps(filters) if any(v for v in filters.values() if v) else 'Not provided'}
"""
    if context:
        user_content += f"\n\nPolicy Information:\n{context}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    # Stream with key rotation
    last_err = None
    for idx, (key, model_name) in enumerate(GEMINI_CANDIDATES):
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key,
                temperature=0.3,
                streaming=True,
            )
            async for chunk in llm.astream(messages):
                text = chunk.content
                if text:
                    yield f"data: {json.dumps({'type': 'answer', 'text': text})}\n\n"
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            err = str(e)
            if any(x in err for x in ["429","quota","503","504","Deadline"]):
                last_err = e; continue
            raise

    yield f"data: {json.dumps({'type': 'error', 'text': f'All models failed: {last_err}'})}\n\n"
    yield "data: [DONE]\n\n"


def _format_sql_result(result: dict, intent: str) -> str:
    candidates = result.get("candidates", [])
    if not candidates:
        return "No matching policies found for the given criteria."
    lines = [f"Found {len(candidates)} matching policies:\n"]
    for p in candidates:
        lines.append(
            f"- **{p.get('insurer')} — {p.get('product_slug')}**: "
            f"SI up to ₹{p.get('si_max_lakhs')}L, "
            f"PED wait {p.get('ped_waiting_months')}m, "
            f"Room rent: {p.get('room_rent_limit')}, "
            f"Co-pay: {p.get('copayment_percent', 0)}%, "
            f"CSR: {p.get('claim_settlement_ratio')}%"
        )
    return "\n".join(lines)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        process_chat(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/health")
def health():
    return {"status": "ok", "policies": len(CORPUS)}

@app.get("/policies")
def list_policies():
    return [{"id": p["id"], "insurer": p["insurer"],
             "slug": p["product_slug"]} for p in CORPUS]

# ── Dev run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
