"""STEP 5: SQL Filter Node"""
import os
from dotenv import load_dotenv
import requests

load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

def _h():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}

def sql_filter(filters: dict, intent: str, limit: int = 10) -> dict:
    params, applied = _build_params(filters, intent, limit)
    r = requests.post(f"{SUPABASE_URL}/rest/v1/rpc/filter_policies", headers=_h(), json=params, timeout=15)
    candidates = r.json() if r.ok else []
    if not r.ok:
        print(f"  ⚠ filter_policies failed ({r.status_code}): {r.text[:100]}")
        result = _fallback_all(limit)
        candidates = result["candidates"]

    # Post-filter: remove policies where user age < min_entry_age
    age = filters.get("age")
    if age and candidates:
        ids = [c["id"] for c in candidates]
        r2  = requests.get(
            f"{SUPABASE_URL}/rest/v1/policies?id=in.({','.join(ids)})&select=id,min_entry_age",
            headers=_h(), timeout=10)
        if r2.ok:
            min_ages   = {row["id"]: (row.get("min_entry_age") or 0) for row in r2.json()}
            candidates = [c for c in candidates if min_ages.get(c["id"], 0) <= age]

    if candidates:
        candidates = _enrich_with_uris(candidates, [c["id"] for c in candidates])

    return {"candidates": candidates, "total_found": len(candidates),
            "filters_applied": applied, "sql_params": params}

def _build_params(filters: dict, intent: str, limit: int):
    params, applied = {"p_limit": limit}, []
    age    = filters.get("age")
    budget = filters.get("budget_yearly_inr")

    if budget:
        adj = int(budget * 0.6) if age and age > 45 else int(budget * 0.8) if age and age > 35 else budget
        params["p_max_premium_yearly"] = adj
        applied.append(f"budget≤₹{budget:,}/yr")

    if age:
        params["p_max_entry_age"] = age
        applied.append(f"age={age}")

    conditions = [c.lower() for c in (filters.get("conditions") or [])]
    if "diabetes" in conditions:
        params["p_diabetes_covered"] = True; applied.append("diabetes_covered")
    if any(c in conditions for c in ["cardiac","heart","coronary"]):
        params["p_cardiac_covered"] = True; applied.append("cardiac_covered")
    if filters.get("maternity_needed"):
        params["p_maternity_required"] = True; applied.append("maternity_required")
    if filters.get("no_copay"):
        params["p_no_copay"] = True; applied.append("no_copay")
    if filters.get("no_room_rent_limit"):
        params["p_no_room_rent_limit"] = True; applied.append("no_room_rent_limit")
    if filters.get("restore_needed"):
        params["p_restore_required"] = True; applied.append("restore_required")
    if filters.get("opd_needed"):
        params["p_opd_required"] = True; applied.append("opd_required")
    if filters.get("insurer"):
        params["p_insurer"] = filters["insurer"]; applied.append(f"insurer={filters['insurer']}")

    if intent == "eligibility_check":
        params = {"p_limit": limit}
        if age: params["p_max_entry_age"] = age
        applied = [f"age={age}"] if age else ["no_filters"]
    if intent == "premium_lookup":
        params = {"p_limit": limit}
        if budget: params["p_max_premium_yearly"] = budget; applied = [f"budget≤₹{budget:,}"]
        if age: params["p_max_entry_age"] = age

    return params, applied

def _enrich_with_uris(candidates, ids):
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/policies?id=in.({','.join(ids)})&select=id,gemini_file_uris",
        headers=_h(), timeout=10)
    if not r.ok: return candidates
    uri_map = {row["id"]: row.get("gemini_file_uris", []) for row in r.json()}
    for c in candidates:
        c["gemini_file_uris"] = uri_map.get(c["id"], [])
    return candidates

def _fallback_all(limit):
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/policies?is_active=eq.true"
        f"&select=id,insurer,product_slug,si_max_lakhs,approx_premium_30yr_5l,"
        f"approx_premium_45yr_10l,ped_waiting_months,room_rent_limit,copayment_percent,"
        f"maternity_covered,restore_benefit,claim_settlement_ratio,feature_score,gemini_file_uris"
        f"&order=feature_score.desc.nullslast&limit={limit}",
        headers=_h(), timeout=10)
    candidates = r.json() if r.ok else []
    return {"candidates": candidates, "total_found": len(candidates),
            "filters_applied": ["fallback_all"], "sql_params": {}}

def get_policy(policy_id):
    r = requests.get(f"{SUPABASE_URL}/rest/v1/policies?id=eq.{policy_id}&select=*&limit=1",
                     headers=_h(), timeout=10)
    return r.json()[0] if r.ok and r.json() else None

def get_policies(policy_ids):
    if not policy_ids: return []
    r = requests.get(f"{SUPABASE_URL}/rest/v1/policies?id=in.({','.join(policy_ids)})&select=*",
                     headers=_h(), timeout=10)
    return r.json() if r.ok else []

if __name__ == "__main__":
    tests = [
        {"label":"36yr gall bladder stone","filters":{"age":36,"conditions":["gall bladder stone"]},"intent":"recommendation"},
        {"label":"maternity no copay","filters":{"maternity_needed":True,"no_copay":True},"intent":"recommendation"},
        {"label":"senior 65yr","filters":{"age":65},"intent":"eligibility_check"},
    ]
    for t in tests:
        r = sql_filter(t["filters"], t["intent"])
        print(f"\n{t['label']}: {[c.get('product_slug') for c in r['candidates']]}")
