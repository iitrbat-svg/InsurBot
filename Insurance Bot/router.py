"""
Router — judgment-based, 4 retrieval decisions instead of 12 rigid intents.
Upgraded to capture explicit premium parameters (Sum Insured and Zones).
"""
import os, json, re
from dotenv import load_dotenv
from langchain_core.prompts      import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai      import ChatGoogleGenerativeAI
import requests as _req
from pathlib import Path

load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2","true")
os.environ.setdefault("LANGCHAIN_PROJECT","insurance-intel")

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

CORPUS      = json.loads(Path("products_v2.json").read_text()) if Path("products_v2.json").exists() else []
CORPUS_LIST = "\n".join(f"  {p['insurer']}: {p['product_slug']} (id: {p['id']})" for p in CORPUS)

ROUTER_SYSTEM = """You are routing queries for an Indian health insurance advisor.
Your job: understand what the user needs, decide how to retrieve the answer, extract key information.

AVAILABLE POLICIES IN OUR DATABASE:
{corpus_list}

CONVERSATION CONTEXT:
{summary}

Return ONLY valid JSON. No markdown, no explanation.

{{
  "retrieval_decision": <ONE of: "NO_RETRIEVAL" | "SQL_ONLY" | "RAG" | "FC">,
  "intent_description": <one sentence describing what user wants — used to guide the synthesizer>,
  "policies_mentioned": [<exact policy ids from corpus if user named specific policies>],
  "insurer_mentioned":  <insurer name if user named insurer but not specific product, else null>,
  "not_in_corpus":      [<policy names user mentioned NOT in our database>],
  "filters": {{
    "age":                <int or null>,
    "budget_yearly_inr":  <int or null — convert "15k"→15000, "1.5L"→150000>,
    "budget_monthly_inr": <int or null>,
    "sum_insured_inr":    <int or null — CRITICAL: If the user mentions a sum insured (e.g., '10 lakhs', '5L', '1 crore'), you MUST convert it to a raw integer. (e.g., 10 lakhs = 1000000, 5L = 500000)>,
    "zone":               <string or null — e.g., "Zone 1", "Zone A", "Tier 1">,
    "si_min_lakhs":       <float or null>,
    "conditions":         [<medical conditions lowercase, e.g. "diabetes","jaundice","gall bladder stone">],
    "maternity_needed":   <bool or null>,
    "no_copay":           <bool or null>,
    "no_room_rent_limit": <bool or null>,
    "opd_needed":         <bool or null>,
    "restore_needed":     <bool or null>,
    "senior_citizen":     <bool or null — true if age >= 60>,
    "city_tier":          <"A"/"B"/"C" or null>
  }},
  "section_tags": [<relevant tags: waiting_period | exclusions | sub_limits | maternity |
                    ncb_restore | claims_process | premium_table | general_coverage | definitions>],
  "specific_question":    <precise retrieval question — specific enough to match policy clauses.
                           NEVER use vague phrases like "key features" or "tell me about".
                           Always ask for specific terms: waiting periods, limits, exclusions, amounts.
                           For cost/premium/comparison questions, ask about specific numerical configurations.
                           For conditions: classify first, then ask accordingly —
                           ACUTE (jaundice/dengue/typhoid/fever/appendicitis/fracture/emergency/infection):
                             ask "What is covered under general hospitalization after 30-day initial wait?"
                           SPECIFIED DISEASE (gall bladder/hernia/cataract/kidney stone/knee replacement/
                             varicose vein/piles/hydrocele/sinusitis/spinal disc/benign tumor/hysterectomy):
                             ask "What is the specified disease waiting period for [condition]?"
                           PRE-EXISTING (diabetes/hypertension/cardiac/declared conditions):
                             ask "What is the PED waiting period and coverage terms for [condition]?">,
  "needs_clarification":  <bool — true ONLY if query is genuinely too vague to answer at all>,
  "clarification_question": <one targeted question if needs_clarification, else null>,
  "is_followup":          <bool — true if references prior context like "that plan","the first one","it">,
  "resolved_policy_ids":  [<if followup, policy ids from prior context>]
}}

CRITICAL PRICING & COMPARISON RULES:
1. If the user asks "Which is cheapest?", "Compare prices", "How much does it cost?", or explicitly requests numerical quotes, you MUST capture their `age` and `sum_insured_inr` inside the `filters` block if identifiable.
2. If they ask to compare specific policies, ensure ALL mentioned policies match target items and are loaded into `policies_mentioned` or `resolved_policy_ids`.
3. If they seek the "cheapest option" across the entire catalog without specifying a policy, set `retrieval_decision` to "SQL_ONLY" to fetch top catalog matches, which the engine will then score and run pricing calculations on.

RETRIEVAL DECISION RULES — use judgment, not rigid matching:

NO_RETRIEVAL: Query answerable from general knowledge or the corpus list alone.
SQL_ONLY: Needs structured fields from policies table or premium engine checks.
RAG: Needs specific clause text from policy documents.
FC (Full Context): User named 1-4 specific policies AND needs deep analysis.
"""

ROUTER_HUMAN = "User message: {message}"

VALID_DECISIONS = {"NO_RETRIEVAL","SQL_ONLY","RAG","FC"}
VALID_TAGS      = {"waiting_period","exclusions","sub_limits","maternity","ncb_restore",
                   "claims_process","premium_table","definitions","general_coverage","other"}

def _make_chain(key, model):
    prompt = ChatPromptTemplate.from_messages([("system",ROUTER_SYSTEM),("human",ROUTER_HUMAN)])
    llm    = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0, max_output_tokens=1024)
    return prompt | llm | StrOutputParser()

def _parse(raw):
    raw = raw.strip()
    # Fixed string matching so chat UI doesn't break the copy-paste
    bt = "`" * 3
    if bt in raw: 
        raw = re.sub(bt + r"(?:json)?", "", raw).strip().rstrip("`").strip()
    s,e = raw.find("{"), raw.rfind("}")
    if s==-1 or e==-1: raise ValueError(f"No JSON: {raw[:100]}")
    return json.loads(raw[s:e+1])

def _validate(r, message):
    if r.get("retrieval_decision") not in VALID_DECISIONS:
        r["retrieval_decision"] = "RAG"
    r.setdefault("policies_mentioned",[])
    r.setdefault("not_in_corpus",[])
    r.setdefault("section_tags",[])
    r.setdefault("resolved_policy_ids",[])
    r.setdefault("filters",{})
    r.setdefault("needs_clarification",False)
    r.setdefault("clarification_question",None)
    r.setdefault("is_followup",False)
    r.setdefault("insurer_mentioned",None)
    r.setdefault("intent_description","")
    r["specific_question"] = r.get("specific_question") or message
    r["section_tags"]      = [t for t in r.get("section_tags",[]) if t in VALID_TAGS]

    if r["retrieval_decision"] == "FC" and not r["policies_mentioned"]:
        r["retrieval_decision"] = "RAG"

    if r.get("not_in_corpus") and not r.get("policies_mentioned"):
        r["retrieval_decision"]  = "NO_RETRIEVAL"
        r["_missing_from_corpus"] = True

    filters = r.get("filters",{})
    is_recommendation = any(w in message.lower() for w in ["best","recommend","suggest","which policy","should i buy"])
    if is_recommendation and not any(v for v in filters.values() if v is not None):
        r["needs_clarification"]    = True
        r["clarification_question"] = ("Could you share your age, approximate annual budget, "
                                       "and any health conditions? That'll help me find the best match.")

    q = r.get("specific_question","")
    vague = ["key features","overview","tell me about","information about","details about"]
    if any(p in q.lower() for p in vague):
        r["specific_question"] = q + " — specifically: waiting periods, exclusions, sub-limits, room rent limit, co-payment."
    return r

def _fallback(message, error):
    return {
        "retrieval_decision":    "RAG",
        "intent_description":    message,
        "policies_mentioned":    [],
        "not_in_corpus":         [],
        "filters":               {},
        "section_tags":          [],
        "specific_question":     message,
        "needs_clarification":   False,
        "clarification_question": None,
        "is_followup":           False,
        "resolved_policy_ids":   [],
        "_error":                str(error),
    }

def route(message, summary="", conversation_history=None):
    eff = summary or "No prior conversation."
    if conversation_history:
        recent  = conversation_history[-2:]
        eff    += "\n\nRecent:\n" + "\n".join(f"{m['role'].upper()}: {m['content'][:200]}" for m in recent)
    inputs   = {"corpus_list":CORPUS_LIST,"summary":eff,"message":message}
    last_err = None
    for key, model_name in GEMINI_CANDIDATES:
        try:
            raw    = _make_chain(key,model_name).invoke(inputs)
            result = _parse(raw)
            return _validate(result, message)
        except Exception as e:
            last_err = e; continue
    return _fallback(message, last_err)