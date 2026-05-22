"""
STEP 4: Router Agent
=====================
Classifies every user message into an intent + extracts structured
filters. Output drives all downstream LangGraph nodes.

Intents:
  single_policy_detail   → FC retrieval on one named policy
  comparison             → FC retrieval on 2-4 named policies
  recommendation         → SQL filter → RAG → ranker
  corpus_search          → RAG across all/filtered products
  condition_specific     → SQL filter + RAG on specific section tags
  premium_lookup         → SQL only
  eligibility_check      → SQL only
  claims_process         → RAG on claims_process section
  evaluation             → assess if a policy is good/suitable
  general_knowledge      → LLM only, no retrieval
  missing_docs           → ask user to upload
  clarification_needed   → ask user for more info
  followup               → resolved reference to prior context

Install:
  pip install langchain langchain-google-genai langgraph
      langsmith python-dotenv

.env:
  GEMINI_KEY_PAID=...
  GEMINI_KEY_1=...
  GEMINI_KEY_2=...
  GEMINI_KEY_3=...
  LANGCHAIN_API_KEY=...
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_PROJECT=insurance-intel
  SUPABASE_URL=...
  SUPABASE_KEY=...
"""

import os, json, re
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT",    "insurance-intel")

from langchain_core.prompts         import ChatPromptTemplate
from langchain_core.output_parsers  import StrOutputParser
from langchain_google_genai         import ChatGoogleGenerativeAI
import requests as _req
from pathlib import Path

# ── Gemini key rotation ────────────────────────────────────────────────────────

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

# ── Corpus index ───────────────────────────────────────────────────────────────

def _load_corpus():
    f = Path("products_v2.json")
    return json.loads(f.read_text()) if f.exists() else []

CORPUS      = _load_corpus()
CORPUS_LIST = "\n".join(
    f"  {p['insurer']}: {p['product_slug']} (id: {p['id']})"
    for p in CORPUS
)

# ── Router prompt ──────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """You are the routing brain of an Indian health insurance comparison system.
Your job: classify the user's query and extract structured information.

AVAILABLE POLICIES IN OUR CORPUS:
{corpus_list}

CONVERSATION SUMMARY (what has been discussed so far):
{summary}

Return ONLY valid JSON. No markdown, no explanation.

{{
  "intent": <one of the intents listed below>,
  "policies_mentioned": [<policy ids from corpus that user named, e.g. ["care_supreme"]>],
  "insurer_mentioned": <insurer name if user named insurer but not specific product, else null>,
  "not_in_corpus": [<policy names user mentioned that are NOT in our corpus>],

  "filters": {{
    "age": <integer or null>,
    "budget_yearly_inr": <integer or null — convert "15k" to 15000>,
    "budget_monthly_inr": <integer or null>,
    "si_min_lakhs": <numeric or null>,
    "conditions": [<medical conditions mentioned, lowercase, e.g. ["diabetes","hypertension"]>],
    "maternity_needed": <true/false/null>,
    "no_copay": <true/false/null>,
    "no_room_rent_limit": <true/false/null>,
    "opd_needed": <true/false/null>,
    "restore_needed": <true/false/null>,
    "senior_citizen": <true/false/null — true if age 60+>,
    "city_tier": <"A"/"B"/"C"/null — A=metro, B=tier1, C=rest>
  }},

  "section_tags": [<relevant retrieval section tags from:
    "waiting_period", "exclusions", "sub_limits", "maternity",
    "ncb_restore", "claims_process", "premium_table",
    "definitions", "general_coverage">],

  "specific_question": <the precise retrieval question — must be specific enough
    to match exact policy clauses, NOT marketing language. Instead of
    "key features and benefits", ask for specific terms like waiting periods,
    sub-limits, exclusions, room rent limit, co-payment, restore benefit etc.>,

  "needs_clarification": <true/false>,
  "clarification_question": <one targeted question if needs_clarification, else null>,
  "is_followup": <true/false — true if references prior context like "that plan" or "the first one">,
  "resolved_policy_ids": [<if followup, policy ids from prior context this refers to>],
  "retrieval_mode": <"full_context"/"rag_filtered"/"sql_only"/"llm_only"/"ask_user">,
  "confidence": <"high"/"medium"/"low">
}}

INTENTS AND HOW TO HANDLE EACH:

  single_policy_detail — user asks a specific question about one named policy.
    specific_question must name exact terms being asked about (waiting period,
    room rent, maternity limit etc.), never use vague phrases like "key features".
    retrieval_mode: full_context

  comparison — user wants 2-4 specific named policies compared on specific terms.
    specific_question should list the exact parameters to compare.
    retrieval_mode: full_context

  recommendation — user gives profile (age/budget/conditions) and wants best policy.
    Extract all filters carefully. If no profile info at all → needs_clarification=true.
    specific_question: describe the user profile and what to optimise for.
    retrieval_mode: rag_filtered

  corpus_search — broad search across all policies.
    Example: "which plans cover OPD from day 1", "which plans have no room rent limit".
    specific_question: the exact policy feature/clause to search for.
    retrieval_mode: rag_filtered

  condition_specific — focused on a medical condition across multiple policies.
    Example: "which plans cover cardiac from day 1", "best plans for diabetics".
    section_tags: always include "waiting_period" and "exclusions".
    retrieval_mode: rag_filtered

  premium_lookup — asking about premium/cost.
    retrieval_mode: sql_only

  eligibility_check — asking about age limits, entry conditions.
    retrieval_mode: sql_only

  claims_process — how to file a claim, cashless process, documents needed.
    section_tags: ["claims_process"]
    retrieval_mode: rag_filtered

  evaluation — user asks if a policy is good/worth it/suitable/recommended.
    Examples: "Is Activ One MAX a good policy?", "Is Care Supreme worth buying?",
    "Should I buy HDFC Optima Secure?", "Is this policy suitable for me?"

    CRITICAL RULES for evaluation:
    - If user gives NO personal profile (age/budget/conditions) →
        needs_clarification=true
        clarification_question="To evaluate if [policy] suits you, could you
        share your age, any health conditions, and approximate annual budget?"
    - If user gives profile → needs_clarification=false
        section_tags: include ALL of: waiting_period, exclusions, sub_limits,
        general_coverage, ncb_restore, maternity
        specific_question must be comprehensive:
        "What are the waiting periods, exclusions, sub-limits, room rent limit,
        co-payment terms, restore benefit, NCB, and key coverage terms of [policy]?
        Retrieve all sections needed for a full suitability evaluation."
    retrieval_mode: rag_filtered (not full_context — we need all sections)

  general_knowledge — generic insurance concepts, no specific product needed.
    Examples: "what is a waiting period", "explain co-payment", "what is NCB".
    retrieval_mode: llm_only

  missing_docs — user asks about a policy NOT in our corpus.
    retrieval_mode: ask_user

  clarification_needed — query too vague to route confidently.
    retrieval_mode: ask_user

  followup — user refers to something from prior conversation.
    Examples: "what about the waiting period in that plan",
    "how does the first option compare on maternity".
    Resolve references using conversation summary.
    retrieval_mode: depends on resolved intent.

RETRIEVAL MODE rules (if not already set by intent):
  full_context  → pass Gemini file URIs directly (best for 1-4 specific policies)
  rag_filtered  → vector search filtered by policy_ids + section_tags
  sql_only      → query structured policies table, no LLM retrieval
  llm_only      → answer from model knowledge, no retrieval needed
  ask_user      → prompt user for missing info or documents
"""

ROUTER_HUMAN = "User message: {message}"

# ── Chain builder ──────────────────────────────────────────────────────────────

def _make_chain(api_key: str, model: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM),
        ("human",  ROUTER_HUMAN),
    ])
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0,
        max_output_tokens=1024,
        convert_system_message_to_human=False,
    )
    return prompt | llm | StrOutputParser()

# ── JSON parser ────────────────────────────────────────────────────────────────

def _parse(raw: str) -> dict:
    raw = raw.strip()
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"No JSON in router output: {raw[:200]}")
    return json.loads(raw[s:e+1])

# ── Validation ─────────────────────────────────────────────────────────────────

VALID_INTENTS = {
    "single_policy_detail", "comparison", "recommendation",
    "corpus_search", "condition_specific", "premium_lookup",
    "eligibility_check", "claims_process", "evaluation",
    "general_knowledge", "missing_docs", "clarification_needed", "followup",
}
VALID_MODES = {
    "full_context", "rag_filtered", "sql_only", "llm_only", "ask_user"
}
VALID_TAGS = {
    "waiting_period", "exclusions", "sub_limits", "maternity",
    "ncb_restore", "claims_process", "premium_table",
    "definitions", "general_coverage", "other",
}
INTENT_DEFAULT_MODE = {
    "single_policy_detail": "full_context",
    "comparison":           "full_context",
    "recommendation":       "rag_filtered",
    "corpus_search":        "rag_filtered",
    "condition_specific":   "rag_filtered",
    "premium_lookup":       "sql_only",
    "eligibility_check":    "sql_only",
    "claims_process":       "rag_filtered",
    "evaluation":           "rag_filtered",
    "general_knowledge":    "llm_only",
    "missing_docs":         "ask_user",
    "clarification_needed": "ask_user",
    "followup":             "rag_filtered",
}
# All section tags needed for a full evaluation
EVALUATION_TAGS = [
    "waiting_period", "exclusions", "sub_limits",
    "general_coverage", "ncb_restore", "maternity",
]

def _validate(result: dict, message: str) -> dict:
    # Intent
    if result.get("intent") not in VALID_INTENTS:
        result["intent"] = "clarification_needed"

    intent = result["intent"]

    # Retrieval mode
    if result.get("retrieval_mode") not in VALID_MODES:
        result["retrieval_mode"] = INTENT_DEFAULT_MODE.get(intent, "rag_filtered")

    # Defaults
    result.setdefault("policies_mentioned",   [])
    result.setdefault("not_in_corpus",        [])
    result.setdefault("section_tags",         [])
    result.setdefault("resolved_policy_ids",  [])
    result.setdefault("filters",              {})
    result.setdefault("needs_clarification",  False)
    result.setdefault("clarification_question", None)
    result.setdefault("is_followup",          False)
    result.setdefault("confidence",           "medium")
    result["specific_question"] = result.get("specific_question") or message
    result.setdefault("insurer_mentioned",    None)

    # Clean section_tags
    result["section_tags"] = [t for t in result["section_tags"] if t in VALID_TAGS]

    # Evaluation-specific rules
    if intent == "evaluation":
        f = result.get("filters", {})
        has_profile = any([
            f.get("age"),
            f.get("budget_yearly_inr"),
            f.get("budget_monthly_inr"),
            f.get("conditions"),
        ])
        if not has_profile:
            result["needs_clarification"] = True
            policies = result.get("policies_mentioned", [])
            policy_name = policies[0].replace("_", " ").title() if policies else "this policy"
            result["clarification_question"] = (
                f"To evaluate if {policy_name} suits you, could you share "
                f"your age, any health conditions, and approximate annual budget?"
            )
        else:
            # Full evaluation — need all section tags
            result["section_tags"]    = EVALUATION_TAGS
            result["retrieval_mode"]  = "rag_filtered"
            # Rewrite specific_question to be comprehensive
            policies  = result.get("policies_mentioned", [])
            pname     = policies[0].replace("_", " ").title() if policies else "this policy"
            result["specific_question"] = (
                f"What are the waiting periods, exclusions, sub-limits, "
                f"room rent limit, co-payment terms, restore benefit, NCB, "
                f"and key coverage terms of {pname}? "
                f"Retrieve all sections needed for a full suitability evaluation."
            )

    # Comparison → single if only 1 policy
    if intent == "comparison" and len(result["policies_mentioned"]) == 1:
        result["intent"]         = "single_policy_detail"
        result["retrieval_mode"] = "full_context"

    # Recommendation with no filters → clarify
    if intent == "recommendation":
        f = result.get("filters", {})
        if not any(v for v in f.values() if v is not None):
            result["needs_clarification"]   = True
            result["clarification_question"] = (
                "Could you share your age, approximate annual budget, "
                "and any health conditions? That'll help me find the best match."
            )

    # not_in_corpus only → missing_docs
    if result.get("not_in_corpus") and not result.get("policies_mentioned"):
        result["intent"]         = "missing_docs"
        result["retrieval_mode"] = "ask_user"

    # Ensure specific_question is never vague for retrieval intents
    if result["retrieval_mode"] in ("full_context", "rag_filtered"):
        q = result.get("specific_question", "")
        vague_phrases = [
            "key features", "benefits", "tell me about",
            "information about", "details about", "overview"
        ]
        if any(p in q.lower() for p in vague_phrases):
            # Append specificity instruction
            result["specific_question"] = (
                q + " Include waiting periods, exclusions, sub-limits, "
                "room rent, co-payment, and restore benefit details."
            )

    return result

def _fallback(message: str, error: str) -> dict:
    return {
        "intent":                 "clarification_needed",
        "retrieval_mode":         "ask_user",
        "policies_mentioned":     [],
        "not_in_corpus":          [],
        "filters":                {},
        "section_tags":           [],
        "specific_question":      message,
        "needs_clarification":    True,
        "clarification_question": "I'm having trouble processing that. Could you rephrase?",
        "is_followup":            False,
        "resolved_policy_ids":    [],
        "confidence":             "low",
        "_error":                 error,
    }

# ── Main route function ────────────────────────────────────────────────────────

def route(
    message: str,
    summary: str = "",
    conversation_history: list | None = None,
) -> dict:
    """
    Route a user message. Returns structured decision dict.

    Args:
        message:              Current user message
        summary:              Rolling conversation summary
        conversation_history: Last 2-3 raw messages for followup resolution
    """
    effective_summary = summary or "No prior conversation."
    if conversation_history:
        recent = conversation_history[-2:]
        recent_str = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}" for m in recent
        )
        effective_summary += f"\n\nRecent messages:\n{recent_str}"

    inputs = {
        "corpus_list": CORPUS_LIST,
        "summary":     effective_summary,
        "message":     message,
    }

    last_error = None
    for key, model_name in GEMINI_CANDIDATES:
        try:
            raw    = _make_chain(key, model_name).invoke(inputs)
            result = _parse(raw)
            return _validate(result, message)
        except Exception as e:
            last_error = e
            continue

    return _fallback(message, str(last_error))

# ── Fuzzy policy name resolver ─────────────────────────────────────────────────

def resolve_policy_name(name: str) -> list[dict]:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        return []
    r = _req.post(
        f"{url}/rest/v1/rpc/resolve_policy_name",
        headers={
            "apikey": key, "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={"query_name": name}, timeout=10,
    )
    return r.json() if r.ok else []

# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    TEST_QUERIES = [
        "Is Activ One MAX a good policy?",
        "Is Care Supreme worth buying? I'm 34, diabetic, budget 18k/year",
        "Is maternity covered in Care Supreme?",
        "I'm 34 years old, diabetic, budget is 15k per year. Which policy is best?",
        "Compare HDFC Optima Secure and Niva Bupa ReAssure on room rent",
        "Which plans have no co-payment clause?",
        "What is a waiting period?",
        "How do I file a cashless claim with Star Health?",
        "What happens to my NCB if I make a claim on Star Comprehensive?",
        "Which plans cover knee replacement surgery?",
        "Tell me about Apollo Munich",
        "What about the waiting period in that plan?",
    ]

    queries = [sys.argv[1]] if len(sys.argv) > 1 else TEST_QUERIES

    print("=" * 65)
    for q in queries:
        print(f"\nQ: {q}")
        r = route(q)
        print(f"  intent:          {r['intent']}")
        print(f"  retrieval_mode:  {r['retrieval_mode']}")
        print(f"  policies:        {r['policies_mentioned']}")
        print(f"  section_tags:    {r['section_tags']}")
        print(f"  question:        {(r.get('specific_question') or '')[:120]}")
        print(f"  filters:         {r['filters']}")
        if r.get("needs_clarification"):
            print(f"  clarify:         {r['clarification_question']}")
        if r.get("not_in_corpus"):
            print(f"  not_in_corpus:   {r['not_in_corpus']}")
        print(f"  confidence:      {r['confidence']}")
    print("\n" + "=" * 65)
