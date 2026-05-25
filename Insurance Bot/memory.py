"""
Conversation Memory Manager
=============================
Stores session state in Supabase conversations table.
Rolling summary updated every SUMMARIZE_EVERY turns.
Each request only sends summary + last N messages = flat token cost.
"""
import os, json, time
from dotenv import load_dotenv
import requests
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

SUPABASE_URL     = os.environ["SUPABASE_URL"]
SUPABASE_KEY     = os.environ["SUPABASE_KEY"]
SUMMARIZE_EVERY  = 4    # summarize after every N turns
KEEP_RAW         = 3    # keep last N raw messages alongside summary
MAX_SUMMARY_TOKENS = 300

GEMINI_CANDIDATES = [(k,m) for k,m in [
    (os.getenv("GEMINI_KEY_PAID"), "gemini-2.5-flash-lite"),
    (os.getenv("GEMINI_KEY_1"),    "gemini-2.0-flash"),
    (os.getenv("GEMINI_KEY_2"),    "gemini-2.0-flash"),
    (os.getenv("GEMINI_KEY_3"),    "gemini-2.0-flash"),
] if k]

def _h():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }

# ── Load session ───────────────────────────────────────────────────────────────

def load_session(session_id: str) -> dict:
    """
    Load session from Supabase. Returns dict with:
      summary, turn_count, recent_messages, user_profile, last_candidate_ids
    If session doesn't exist, returns empty defaults.
    """
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/conversations?id=eq.{session_id}&select=*&limit=1",
        headers=_h(), timeout=10)

    if r.ok and r.json():
        row = r.json()[0]
        return {
            "summary":           row.get("summary") or "",
            "turn_count":        row.get("turn_count") or 0,
            "recent_messages":   row.get("user_profile", {}).get("_recent_messages", []),
            "user_profile":      {k:v for k,v in (row.get("user_profile") or {}).items()
                                  if not k.startswith("_")},
            "last_candidate_ids": row.get("last_candidate_ids") or [],
        }
    return {
        "summary":           "",
        "turn_count":        0,
        "recent_messages":   [],
        "user_profile":      {},
        "last_candidate_ids": [],
    }

# ── Save session ───────────────────────────────────────────────────────────────

def save_session(session_id: str, session: dict):
    """Upsert session state to Supabase."""
    # Pack recent_messages into user_profile JSONB (avoids schema change)
    profile = dict(session.get("user_profile") or {})
    profile["_recent_messages"] = session.get("recent_messages", [])

    row = {
        "id":                 session_id,
        "summary":            session.get("summary", ""),
        "turn_count":         session.get("turn_count", 0),
        "last_candidate_ids": session.get("last_candidate_ids", []),
        "user_profile":       profile,
        "updated_at":         "now()",
    }
    requests.post(
        f"{SUPABASE_URL}/rest/v1/conversations",
        headers={**_h(), "Prefer": "resolution=merge-duplicates"},
        json=row, timeout=10)

# ── Add turn ───────────────────────────────────────────────────────────────────

def add_turn(session: dict, user_msg: str, ai_msg: str) -> dict:
    """
    Add a user+AI turn to session.
    Trims recent_messages to last KEEP_RAW turns.
    Increments turn_count.
    """
    msgs = session.get("recent_messages", [])
    msgs.append({"role": "user",      "content": user_msg})
    msgs.append({"role": "assistant", "content": ai_msg[:500]})  # cap AI msg length
    session["recent_messages"] = msgs[-(KEEP_RAW * 2):]  # keep last N turns
    session["turn_count"]      = session.get("turn_count", 0) + 1
    return session

# ── Update user profile ────────────────────────────────────────────────────────

def update_profile(session: dict, filters: dict) -> dict:
    """
    Accumulate user profile across turns.
    E.g. if user mentions age in turn 1 and budget in turn 3,
    both are available in turn 4.
    """
    profile = session.get("user_profile", {})
    for key in ["age","budget_yearly_inr","budget_monthly_inr","city_tier"]:
        if filters.get(key) is not None:
            profile[key] = filters[key]
    # Merge conditions (deduplicate)
    existing   = set(profile.get("conditions", []))
    new_conds  = set(filters.get("conditions") or [])
    if new_conds:
        profile["conditions"] = list(existing | new_conds)
    # Boolean preferences
    for key in ["maternity_needed","no_copay","no_room_rent_limit",
                "opd_needed","restore_needed","senior_citizen"]:
        if filters.get(key) is not None:
            profile[key] = filters[key]
    session["user_profile"] = profile
    return session

# ── Summarize ─────────────────────────────────────────────────────────────────

def should_summarize(session: dict) -> bool:
    return session.get("turn_count", 0) % SUMMARIZE_EVERY == 0 and \
           session.get("turn_count", 0) > 0

def summarize(session: dict) -> str:
    """
    Compress conversation into a short summary using cheapest Gemini model.
    Called every SUMMARIZE_EVERY turns. Costs ~200-400 tokens per call.
    """
    msgs    = session.get("recent_messages", [])
    profile = session.get("user_profile", {})
    prior   = session.get("summary", "")

    if not msgs:
        return prior

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}" for m in msgs)

    prompt = f"""Summarize this health insurance conversation in 3 sentences max.
Include: user profile (age/conditions/budget if mentioned), policies discussed,
key findings, and what the user still needs to know.
Be factual and specific. No filler.

Prior summary: {prior or 'None'}

Recent conversation:
{history_text}

User profile so far: {json.dumps(profile)}

Output ONLY the summary. No preamble."""

    last_err = None
    for key, model_name in GEMINI_CANDIDATES:
        try:
            llm  = ChatGoogleGenerativeAI(model=model_name, google_api_key=key,
                                          temperature=0, max_output_tokens=MAX_SUMMARY_TOKENS)
            resp = llm.invoke([HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception as e:
            last_err = e; continue

    # Fallback: keep prior summary if Gemini fails
    print(f"  ⚠ Summarization failed: {last_err}")
    return prior or ""

# ── Enrich router filters with session profile ─────────────────────────────────

def enrich_filters(filters: dict, session: dict) -> dict:
    """
    Fill in missing filters from accumulated session profile.
    E.g. user asked about maternity in turn 1, asks about budget in turn 2
    — turn 2 query should still know about maternity.
    """
    profile  = session.get("user_profile", {})
    enriched = dict(filters)
    for key in ["age","budget_yearly_inr","city_tier"]:
        if enriched.get(key) is None and profile.get(key) is not None:
            enriched[key] = profile[key]
    # Merge conditions
    existing = set(enriched.get("conditions") or [])
    profile_conds = set(profile.get("conditions") or [])
    enriched["conditions"] = list(existing | profile_conds) or None
    return enriched

# ── Resolve follow-up references ───────────────────────────────────────────────

def resolve_followup(router_result: dict, session: dict) -> dict:
    """
    If router detects a follow-up ("that policy", "the first one"),
    fill resolved_policy_ids from last_candidate_ids in session.
    """
    if not router_result.get("is_followup"):
        return router_result
    if not router_result.get("resolved_policy_ids"):
        router_result["resolved_policy_ids"] = session.get("last_candidate_ids", [])
        if router_result["resolved_policy_ids"]:
            router_result["policies_mentioned"] = router_result["resolved_policy_ids"]
    return router_result
