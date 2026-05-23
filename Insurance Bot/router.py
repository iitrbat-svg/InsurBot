"""STEP 4: Router Agent"""
import os, json, re
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2","true")
os.environ.setdefault("LANGCHAIN_PROJECT","insurance-intel")

from langchain_core.prompts      import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai      import ChatGoogleGenerativeAI
import requests as _req
from pathlib import Path

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

CORPUS = json.loads(Path("products_v2.json").read_text()) if Path("products_v2.json").exists() else []
CORPUS_LIST = "\n".join(f"  {p['insurer']}: {p['product_slug']} (id: {p['id']})" for p in CORPUS)

SPECIFIED_DISEASES = [
    "gall bladder","bile duct","gallstone","kidney stone","calculi","hernia",
    "cataract","knee replacement","joint replacement","tonsil","adenoid",
    "hysterectomy","benign tumor","benign tumour","varicose vein","piles",
    "fissure","fistula","hydrocele","sinusitis","spinal disc","prolapse disc",
    "pilonidal","mastoid","prostate hypertrophy","gastric ulcer","duodenal ulcer",
    "osteoarthritis","osteoporosis",
]
ACUTE_ILLNESSES = [
    "jaundice","dengue","typhoid","malaria","fever","appendicitis","fracture",
    "accident","injury","pneumonia","infection","flu","covid","viral","bacterial",
    "food poisoning","diarrhea","vomiting","stroke","heart attack","emergency",
]

ROUTER_SYSTEM = """You are routing queries for an Indian health insurance advisor.

AVAILABLE POLICIES:
{corpus_list}

CONVERSATION SUMMARY:
{summary}

Return ONLY valid JSON. No markdown.

{{
  "intent": <see intents below>,
  "policies_mentioned": [<policy ids from corpus>],
  "insurer_mentioned": <insurer name or null>,
  "not_in_corpus": [<policy names not in corpus>],
  "filters": {{
    "age": <int or null>,
    "budget_yearly_inr": <int or null — convert "15k"→15000>,
    "budget_monthly_inr": <int or null>,
    "si_min_lakhs": <float or null>,
    "conditions": [<medical conditions lowercase>],
    "maternity_needed": <bool or null>,
    "no_copay": <bool or null>,
    "no_room_rent_limit": <bool or null>,
    "opd_needed": <bool or null>,
    "restore_needed": <bool or null>,
    "senior_citizen": <bool or null>,
    "city_tier": <"A"/"B"/"C" or null>
  }},
  "section_tags": [<from: waiting_period,exclusions,sub_limits,maternity,ncb_restore,claims_process,premium_table,definitions,general_coverage>],
  "specific_question": <precise retrieval question — never use vague phrases like "key features". For conditions always ask specifically about waiting periods, coverage terms, exclusions.
  CONDITION CLASSIFICATION — include in specific_question:
  - Acute illness (jaundice,dengue,typhoid,fever,appendicitis,fracture,emergency): ask about "general hospitalization coverage and initial 30-day waiting period"
  - Specified disease (gall bladder,hernia,cataract,kidney stone,knee replacement): ask about "specific disease waiting period"
  - Pre-existing (diabetes,hypertension,cardiac): ask about "PED waiting period and coverage terms">,
  "needs_clarification": <bool>,
  "clarification_question": <string or null>,
  "is_followup": <bool>,
  "resolved_policy_ids": [<policy ids from prior context if followup>],
  "retrieval_mode": <"full_context"/"rag_filtered"/"sql_only"/"llm_only"/"ask_user">,
  "confidence": <"high"/"medium"/"low">
}}

INTENTS:
  single_policy_detail — one specific policy, specific question
  comparison           — 2-4 named policies compared
  recommendation       — user profile given, find best policy
  corpus_search        — search across all policies for a feature
  condition_specific   — condition across multiple policies
  premium_lookup       — cost/premium question → sql_only
  eligibility_check    — age/entry conditions → sql_only
  claims_process       — how to file a claim
  evaluation           — is this policy good/suitable
  general_knowledge    — insurance concepts, no retrieval → llm_only
  missing_docs         — policy not in corpus → ask_user
  clarification_needed — too vague → ask_user
  followup             — references prior conversation

RETRIEVAL MODE:
  single_policy_detail/comparison/evaluation → full_context
  recommendation/condition_specific/corpus_search → rag_filtered
  premium_lookup/eligibility_check → sql_only
  general_knowledge → llm_only
  missing_docs/clarification_needed → ask_user

EVALUATION RULE: "is X good/worth it/suitable?" with no profile → needs_clarification=true.
With profile → section_tags=all 6 sections, comprehensive specific_question.

VAGUE QUESTION RULE: specific_question must never contain "key features","overview","tell me about","information about". Always ask for specific terms."""

ROUTER_HUMAN = "User message: {message}"

VALID_INTENTS = {
    "single_policy_detail","comparison","recommendation","corpus_search",
    "condition_specific","premium_lookup","eligibility_check","claims_process",
    "evaluation","general_knowledge","missing_docs","clarification_needed","followup",
}
VALID_MODES   = {"full_context","rag_filtered","sql_only","llm_only","ask_user"}
VALID_TAGS    = {"waiting_period","exclusions","sub_limits","maternity","ncb_restore",
                 "claims_process","premium_table","definitions","general_coverage","other"}
INTENT_MODE   = {
    "single_policy_detail":"full_context","comparison":"full_context",
    "recommendation":"rag_filtered","corpus_search":"rag_filtered",
    "condition_specific":"rag_filtered","premium_lookup":"sql_only",
    "eligibility_check":"sql_only","claims_process":"rag_filtered",
    "evaluation":"rag_filtered","general_knowledge":"llm_only",
    "missing_docs":"ask_user","clarification_needed":"ask_user","followup":"rag_filtered",
}
EVAL_TAGS = ["waiting_period","exclusions","sub_limits","general_coverage","ncb_restore","maternity"]

def _make_chain(key, model):
    prompt = ChatPromptTemplate.from_messages([("system",ROUTER_SYSTEM),("human",ROUTER_HUMAN)])
    llm    = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0, max_output_tokens=1024)
    return prompt | llm | StrOutputParser()

def _parse(raw):
    raw = raw.strip()
    if "```" in raw: raw = re.sub(r"```(?:json)?","",raw).strip().rstrip("`").strip()
    s,e = raw.find("{"), raw.rfind("}")
    if s==-1 or e==-1: raise ValueError(f"No JSON: {raw[:100]}")
    return json.loads(raw[s:e+1])

def _validate(r, message):
    if r.get("intent") not in VALID_INTENTS: r["intent"] = "clarification_needed"
    intent = r["intent"]
    if r.get("retrieval_mode") not in VALID_MODES: r["retrieval_mode"] = INTENT_MODE.get(intent,"rag_filtered")
    r.setdefault("policies_mentioned",[])
    r.setdefault("not_in_corpus",[])
    r.setdefault("section_tags",[])
    r.setdefault("resolved_policy_ids",[])
    r.setdefault("filters",{})
    r.setdefault("needs_clarification",False)
    r.setdefault("clarification_question",None)
    r.setdefault("is_followup",False)
    r.setdefault("confidence","medium")
    r.setdefault("insurer_mentioned",None)
    r["specific_question"] = r.get("specific_question") or message
    r["section_tags"] = [t for t in r["section_tags"] if t in VALID_TAGS]

    # Condition classification — override section_tags and specific_question
    conditions = [c.lower() for c in (r.get("filters",{}).get("conditions") or [])]
    if conditions:
        joined = " ".join(conditions)
        if any(d in joined for d in SPECIFIED_DISEASES):
            if "waiting_period" not in r["section_tags"]: r["section_tags"].insert(0,"waiting_period")
            if "sub_limits"     not in r["section_tags"]: r["section_tags"].append("sub_limits")
        elif any(d in joined for d in ACUTE_ILLNESSES):
            r["section_tags"]       = ["general_coverage","claims_process"]
            r["specific_question"]  = ("What is covered under general hospitalization and acute illness? "
                                       "What is the initial 30-day waiting period and cashless facility?")

    # Evaluation rules
    if intent == "evaluation":
        f = r.get("filters",{})
        if not any([f.get("age"),f.get("budget_yearly_inr"),f.get("conditions")]):
            r["needs_clarification"]    = True
            pids = r.get("policies_mentioned",[])
            pname = pids[0].replace("_"," ").title() if pids else "this policy"
            r["clarification_question"] = (f"To evaluate if {pname} suits you, could you share "
                                           f"your age, any health conditions, and annual budget?")
        else:
            r["section_tags"]       = EVAL_TAGS
            r["retrieval_mode"]     = "rag_filtered"
            pids  = r.get("policies_mentioned",[])
            pname = pids[0].replace("_"," ").title() if pids else "this policy"
            r["specific_question"]  = (f"What are the waiting periods, exclusions, sub-limits, "
                                       f"room rent, co-payment, restore benefit, NCB of {pname}?")

    # Comparison → single if only 1 policy
    if intent == "comparison" and len(r["policies_mentioned"]) == 1:
        r["intent"] = "single_policy_detail"; r["retrieval_mode"] = "full_context"

    # Recommendation with no filters → clarify
    if intent == "recommendation" and not any(v for v in r.get("filters",{}).values() if v is not None):
        r["needs_clarification"]    = True
        r["clarification_question"] = ("Could you share your age, approximate annual budget, "
                                       "and any health conditions?")

    # not_in_corpus only → missing_docs
    if r.get("not_in_corpus") and not r.get("policies_mentioned"):
        r["intent"] = "missing_docs"; r["retrieval_mode"] = "ask_user"

    # Vague question guard
    q = r.get("specific_question","")
    if any(p in q.lower() for p in ["key features","overview","tell me about","information about"]):
        r["specific_question"] = q + " Include waiting periods, exclusions, sub-limits, room rent, co-payment."

    return r

def _fallback(message, error):
    return {"intent":"clarification_needed","retrieval_mode":"ask_user",
            "policies_mentioned":[],"not_in_corpus":[],"filters":{},
            "section_tags":[],"specific_question":message,
            "needs_clarification":True,
            "clarification_question":"I'm having trouble processing that. Could you rephrase?",
            "is_followup":False,"resolved_policy_ids":[],"confidence":"low","_error":error}

def route(message, summary="", conversation_history=None):
    eff_summary = summary or "No prior conversation."
    if conversation_history:
        recent  = conversation_history[-2:]
        eff_summary += "\n\nRecent:\n" + "\n".join(f"{m['role'].upper()}: {m['content'][:200]}" for m in recent)
    inputs = {"corpus_list":CORPUS_LIST,"summary":eff_summary,"message":message}
    last_err = None
    for key, model_name in GEMINI_CANDIDATES:
        try:
            raw    = _make_chain(key, model_name).invoke(inputs)
            result = _parse(raw)
            return _validate(result, message)
        except Exception as e:
            last_err = e; continue
    return _fallback(message, str(last_err))

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
        "Is Activ One MAX a good policy?",
        "I am 36 with jaundice, which policy covers me from year 1 with cashless?",
        "I am 36, have gall bladder stone, which policy covers surgery from year 1?",
        "I am 34, diabetic, budget 15k/year. Best policy?",
        "Compare HDFC Optima Secure vs Niva Bupa ReAssure on room rent",
        "Which plans have no room rent limit?",
        "What is a waiting period?",
        "How to file cashless claim with Star Health?",
    ]
    queries = [sys.argv[1]] if len(sys.argv)>1 else TESTS
    print("="*65)
    for q in queries:
        r = route(q)
        print(f"\nQ: {q}")
        print(f"  intent:    {r['intent']}")
        print(f"  mode:      {r['retrieval_mode']}")
        print(f"  tags:      {r['section_tags']}")
        print(f"  question:  {(r.get('specific_question') or '')[:100]}")
        print(f"  filters:   {r['filters']}")
        if r.get("needs_clarification"): print(f"  clarify:   {r['clarification_question']}")
    print("="*65)
