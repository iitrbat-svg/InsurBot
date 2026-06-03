"""
Router — judgment-based, 4 retrieval decisions instead of 12 rigid intents.
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

RETRIEVAL DECISION RULES — use judgment, not rigid matching:

NO_RETRIEVAL: Query answerable from general knowledge or the corpus list alone.
  Examples: "what is a waiting period", "how does NCB work", "what is co-payment",
  "what policies do you have", "what insurers are covered", "tell me about all policies",
  "what is health insurance", "explain restore benefit", "what is TPA".
  Use this whenever no specific policy clause lookup is needed.

SQL_ONLY: Needs structured fields from policies table only — no document text needed.
  Examples: "which policies allow entry at age 68", "cheapest policy for 30yr 5L SI",
  "which plans have no copay", "policies with max SI above 1Cr", "which cover maternity".
  Use when the answer comes from policy metadata fields, not from clause text.

RAG: Needs specific clause text from policy documents.
  Examples: "what is the maternity waiting period in Care Supreme", 
  "which plans cover OPD from day 1", "what are the exclusions in HDFC Optima",
  "I am 36 with jaundice which policy covers me", "best plan for diabetic 35yr budget 15k",
  "compare room rent across all policies", "which plans cover gall bladder surgery".
  Use for most specific coverage questions, recommendations, condition-specific queries.

FC (Full Context): User named 1-4 specific policies AND needs deep analysis.
  Examples: "compare Star Comprehensive vs Care Supreme on maternity",
  "evaluate HDFC Optima Secure for my profile", "what does Activ One MAX cover for cardiac",
  "is Niva Bupa ReAssure 3.0 worth buying for a 40yr diabetic".
  Use ONLY when specific named policies need detailed document-level analysis.

SECTION TAG GUIDANCE:
  Condition questions        → waiting_period + relevant tag
  Exclusion questions        → exclusions
  Limit/sub-limit questions  → sub_limits
  Maternity questions        → maternity + waiting_period
  Claim questions            → claims_process
  Premium/cost questions     → premium_table
  NCB/restore questions      → ncb_restore
  General coverage           → general_coverage

CLARIFICATION: Only ask for clarification if the query has NO extractable information.
  "Which is the best policy?" → needs_clarification=true (no profile at all)
  "I am 36 with jaundice, best policy?" → needs_clarification=false (enough info to answer)
  "Is this policy good?" with no policy named → needs_clarification=true"""

ROUTER_HUMAN = "User message: {message}"

VALID_DECISIONS = {"NO_RETRIEVAL","SQL_ONLY","RAG","FC"}
VALID_TAGS      = {"waiting_period","exclusions","sub_limits","maternity","ncb_restore",
                   "claims_process","premium_table","definitions","general_coverage","other"}

def _make_chain(key, model):
    prompt = ChatPromptTemplate.from_messages([("system",ROUTER_SYSTEM),("human",ROUTER_HUMAN)])
    llm    = ChatGoogleGenerativeAI(model=model, google_api_key=key,
                                    temperature=0, max_output_tokens=1024)
    return prompt | llm | StrOutputParser()

def _parse(raw):
    raw = raw.strip()
    if "```" in raw: raw = re.sub(r"```(?:json)?","",raw).strip().rstrip("`").strip()
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

    # FC requires named policies — downgrade to RAG if none
    if r["retrieval_decision"] == "FC" and not r["policies_mentioned"]:
        r["retrieval_decision"] = "RAG"

    # not_in_corpus only → flag it
    if r.get("not_in_corpus") and not r.get("policies_mentioned"):
        r["retrieval_decision"]  = "NO_RETRIEVAL"
        r["_missing_from_corpus"] = True

    # No filters at all + RAG → might need clarification for recommendation
    filters = r.get("filters",{})
    is_recommendation = any(w in message.lower() for w in
                            ["best","recommend","suggest","which policy","should i buy"])
    if is_recommendation and not any(v for v in filters.values() if v is not None):
        r["needs_clarification"]    = True
        r["clarification_question"] = ("Could you share your age, approximate annual budget, "
                                       "and any health conditions? That'll help me find the best match.")

    # Vague question guard
    q = r.get("specific_question","")
    vague = ["key features","overview","tell me about","information about","details about"]
    if any(p in q.lower() for p in vague):
        r["specific_question"] = q + (" — specifically: waiting periods, "
                                      "exclusions, sub-limits, room rent limit, co-payment.")
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
        eff    += "\n\nRecent:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}" for m in recent)
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

def resolve_policy_name(name):
    url,key = os.environ.get("SUPABASE_URL",""), os.environ.get("SUPABASE_KEY","")
    if not url or not key: return []
    r = _req.post(f"{url}/rest/v1/rpc/resolve_policy_name",
        headers={"apikey":key,"Authorization":f"Bearer {key}","Content-Type":"application/json"},
        json={"query_name":name}, timeout=10)
    return r.json() if r.ok else []

if __name__ == "__main__":
    import sys
    TESTS = [
        "Tell me about all health insurance policies in India",
        "What is a waiting period?",
        "I am 36 with jaundice, which policy covers me from year 1 with cashless?",
        "I am 36, have gall bladder stone, which policy covers surgery from year 1?",
        "I am 34, diabetic, budget 15k/year. Best policy?",
        "Compare HDFC Optima Secure vs Niva Bupa ReAssure on room rent",
        "Which plans have no room rent limit?",
        "Is Activ One MAX a good policy?",
        "Is Activ One MAX good for me? I am 42, cardiac history, budget 25k",
        "How to file cashless claim with Star Health?",
        "Which is the best policy?",
    ]
    queries = [sys.argv[1]] if len(sys.argv)>1 else TESTS
    print("="*65)
    for q in queries:
        r = route(q)
        print(f"\nQ: {q}")
        print(f"  decision:    {r['retrieval_decision']}")
        print(f"  intent:      {r['intent_description']}")
        print(f"  tags:        {r['section_tags']}")
        print(f"  question:    {(r.get('specific_question') or '')[:100]}")
        print(f"  policies:    {r['policies_mentioned']}")
        filters = {k:v for k,v in r['filters'].items() if v is not None}
        if filters: print(f"  filters:     {filters}")
        if r.get("needs_clarification"): print(f"  clarify:     {r['clarification_question']}")
    print("="*65)
