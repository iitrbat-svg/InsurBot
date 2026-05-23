"""
STEP 5: SQL Filter Node
========================
Takes structured filters from the Router and queries the Supabase
policies table. Returns candidate policy IDs + metadata.

No LLM. Pure SQL via Supabase REST API.
Used by: recommendation, corpus_search, condition_specific,
         premium_lookup, eligibility_check intents.

.env: SUPABASE_URL, SUPABASE_KEY
"""

import os, json
from typing import Optional
from dotenv import load_dotenv
import requests

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

# ── Headers ────────────────────────────────────────────────────────────────────

def _h():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }

# ── Main filter function ───────────────────────────────────────────────────────

def sql_filter(
    filters: dict,
    intent: str,
    limit: int = 10,
) -> dict:
    """
    Query policies table using structured filters from router output.

    Args:
        filters: dict from router — age, budget_yearly_inr, conditions etc.
        intent:  router intent — affects which filters are applied
        limit:   max candidates to return

    Returns:
        {
          "candidates": [...],     # list of policy dicts
          "total_found": int,
          "filters_applied": [...], # which filters were active
          "sql_params": {...},      # params sent to Supabase function
        }
    """
    params, applied = _build_params(filters, intent, limit)

    # Call the filter_policies SQL function from Step 1 schema
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/filter_policies",
        headers=_h(),
        json=params,
        timeout=15,
    )

    if not r.ok:
        # Fallback: return all active policies if filter fails
        print(f"  ⚠ filter_policies RPC failed ({r.status_code}): {r.text[:100]}")
        return _fallback_all(limit)

    candidates = r.json()

    # Post-process: add gemini_file_uris for FC retrieval
    if candidates:
        ids = [c["id"] for c in candidates]
        candidates = _enrich_with_uris(candidates, ids)

    return {
        "candidates":      candidates,
        "total_found":     len(candidates),
        "filters_applied": applied,
        "sql_params":      params,
    }


def _build_params(filters: dict, intent: str, limit: int) -> tuple[dict, list]:
    """
    Map router filters → filter_policies RPC params.
    Returns (params_dict, list_of_applied_filter_names).
    """
    params  = {"p_limit": limit}
    applied = []

    age    = filters.get("age")
    budget = filters.get("budget_yearly_inr")

    # Budget → max premium
    # We use approx_premium_30yr_5l as a proxy for filtering
    # If age > 45, use approx_premium_45yr_10l comparison instead
    # (SQL function uses 30yr/5L as the filter column — add 20% buffer for age)
    if budget:
        # Apply buffer: actual premium for older/higher SI may be higher
        # so we give 30% headroom
        if age and age > 45:
            budget_adjusted = int(budget * 0.6)   # 45yr premium ~1.7x 30yr
        elif age and age > 35:
            budget_adjusted = int(budget * 0.8)
        else:
            budget_adjusted = budget
        params["p_max_premium_yearly"] = budget_adjusted
        applied.append(f"budget≤₹{budget:,}/yr")

    # Age → eligibility
    if age:
        params["p_max_entry_age"] = age
        applied.append(f"age={age}")

    # Conditions
    conditions = [c.lower() for c in (filters.get("conditions") or [])]
    if "diabetes" in conditions:
        params["p_diabetes_covered"] = True
        applied.append("diabetes_covered")
    if any(c in conditions for c in ["hypertension", "bp", "blood pressure"]):
        # hypertension not a direct filter — covered under PED generically
        # handled by RAG, not SQL
        pass
    if any(c in conditions for c in ["cardiac", "heart", "coronary"]):
        params["p_cardiac_covered"] = True
        applied.append("cardiac_covered")

    # Maternity
    if filters.get("maternity_needed"):
        params["p_maternity_required"] = True
        applied.append("maternity_required")

    # No copay
    if filters.get("no_copay"):
        params["p_no_copay"] = True
        applied.append("no_copay")

    # No room rent limit
    if filters.get("no_room_rent_limit"):
        params["p_no_room_rent_limit"] = True
        applied.append("no_room_rent_limit")

    # Restore
    if filters.get("restore_needed"):
        params["p_restore_required"] = True
        applied.append("restore_required")

    # OPD
    if filters.get("opd_needed"):
        params["p_opd_required"] = True
        applied.append("opd_required")

    # Senior citizen → use senior-specific products
    if filters.get("senior_citizen") or (age and age >= 60):
        # Don't filter by age here — let max_entry_age handle it
        # But increase limit to ensure senior products included
        params["p_max_entry_age"] = age or 65
        applied.append("senior_eligibility")

    # Insurer filter (if user named specific insurer)
    # Passed separately from router's insurer_mentioned field
    # Caller can pass insurer via filters["insurer"] if needed
    if filters.get("insurer"):
        params["p_insurer"] = filters["insurer"]
        applied.append(f"insurer={filters['insurer']}")

    # Intent-specific overrides
    if intent == "eligibility_check":
        # Only apply age filter, ignore others
        params = {"p_limit": limit}
        if age:
            params["p_max_entry_age"] = age
        applied = [f"age={age}"] if age else ["no_filters"]

    if intent == "premium_lookup":
        # Only apply age + budget
        params = {"p_limit": limit}
        if budget:
            params["p_max_premium_yearly"] = budget
            applied = [f"budget≤₹{budget:,}"]
        if age:
            params["p_max_entry_age"] = age

    return params, applied


def _enrich_with_uris(candidates: list, ids: list) -> list:
    """
    Fetch gemini_file_uris for the candidate policy IDs.
    Needed by FC Retriever node.
    """
    ids_str = ",".join(f'"{i}"' for i in ids)
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/policies"
        f"?id=in.({','.join(ids)})"
        f"&select=id,gemini_file_uris",
        headers=_h(),
        timeout=10,
    )
    if not r.ok:
        return candidates   # return without URIs if fetch fails

    uri_map = {row["id"]: row.get("gemini_file_uris", []) for row in r.json()}
    for c in candidates:
        c["gemini_file_uris"] = uri_map.get(c["id"], [])
    return candidates


def _fallback_all(limit: int) -> dict:
    """Return top policies by feature_score when filter RPC fails."""
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/policies"
        f"?is_active=eq.true"
        f"&select=id,insurer,product_slug,si_max_lakhs,"
        f"approx_premium_30yr_5l,approx_premium_45yr_10l,"
        f"ped_waiting_months,room_rent_limit,copayment_percent,"
        f"maternity_covered,restore_benefit,claim_settlement_ratio,"
        f"feature_score,gemini_file_uris"
        f"&order=feature_score.desc.nullslast"
        f"&limit={limit}",
        headers=_h(),
        timeout=10,
    )
    if not r.ok:
        return {"candidates": [], "total_found": 0,
                "filters_applied": [], "sql_params": {}}
    candidates = r.json()
    return {
        "candidates":      candidates,
        "total_found":     len(candidates),
        "filters_applied": ["fallback_all"],
        "sql_params":      {},
    }

#── Post-filter: remove policies where user age is below minimum entry age ───────
if filters.get("age"):
    age = filters["age"]
    # Fetch min_entry_age for candidates
    ids = [c["id"] for c in candidates]
    if ids:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/policies"
            f"?id=in.({','.join(ids)})"
            f"&select=id,min_entry_age",
            headers=_h(), timeout=10)
        if r.ok:
            min_ages = {row["id"]: row.get("min_entry_age", 0) for row in r.json()}
            candidates = [
                c for c in candidates
                if min_ages.get(c["id"], 0) <= age
            ]



# ── Fetch single policy ────────────────────────────────────────────────────────

def get_policy(policy_id: str) -> Optional[dict]:
    """Fetch full policy row by ID. Used by FC Retriever."""
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/policies"
        f"?id=eq.{policy_id}"
        f"&select=*"
        f"&limit=1",
        headers=_h(),
        timeout=10,
    )
    if not r.ok or not r.json():
        return None
    return r.json()[0]


def get_policies(policy_ids: list[str]) -> list[dict]:
    """Fetch multiple policy rows by ID list."""
    if not policy_ids:
        return []
    ids_csv = ",".join(policy_ids)
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/policies"
        f"?id=in.({ids_csv})"
        f"&select=*",
        headers=_h(),
        timeout=10,
    )
    if not r.ok:
        return []
    return r.json()


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    TEST_CASES = [
        {
            "label":   "34yr diabetic, budget 15k",
            "filters": {"age": 34, "budget_yearly_inr": 15000,
                        "conditions": ["diabetes"]},
            "intent":  "recommendation",
        },
        {
            "label":   "maternity needed, no copay",
            "filters": {"maternity_needed": True, "no_copay": True},
            "intent":  "recommendation",
        },
        {
            "label":   "senior citizen, age 65",
            "filters": {"age": 65},
            "intent":  "eligibility_check",
        },
        {
            "label":   "no room rent limit",
            "filters": {"no_room_rent_limit": True},
            "intent":  "corpus_search",
        },
        {
            "label":   "cardiac covered, budget 20k",
            "filters": {"conditions": ["cardiac"], "budget_yearly_inr": 20000},
            "intent":  "condition_specific",
        },
        {
            "label":   "no filters (fallback test)",
            "filters": {},
            "intent":  "recommendation",
        },
    ]

    print("=" * 65)
    for tc in TEST_CASES:
        print(f"\nTest: {tc['label']}")
        result = sql_filter(tc["filters"], tc["intent"])
        print(f"  Filters applied: {result['filters_applied']}")
        print(f"  Candidates ({result['total_found']}):")
        for c in result["candidates"]:
            slug    = c.get("product_slug", c.get("id"))
            insurer = c.get("insurer", "")
            prem    = c.get("approx_premium_30yr_5l", "?")
            ped     = c.get("ped_waiting_months", "?")
            rr      = c.get("room_rent_limit", "?")
            print(f"    {insurer:<22} {slug:<30} "
                  f"₹{prem or '?':>6}  PED:{ped}m  RR:{rr}")
    print("\n" + "=" * 65)
