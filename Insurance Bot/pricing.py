"""
Pricing Calculation Module with Fuzzy Age Matching and Closest SI & Zone Fallbacks.
"""
import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

def _h():
    return {
        "apikey": SUPABASE_KEY, 
        "Authorization": f"Bearer {SUPABASE_KEY}", 
        "Content-Type": "application/json"
    }

def matches_age(band_str, age):
    """
    Intelligently parses messy age band strings.
    """
    if not band_str: return False
    text = str(band_str).lower().strip()
    
    nums = [int(n) for n in re.findall(r'\d+', text)]
    
    if len(nums) >= 2:
        return nums[0] <= age <= nums[1]
    elif len(nums) == 1:
        if 'up to' in text or '<' in text or 'upto' in text:
            return age <= nums[0]
        if '>' in text or 'above' in text or '+' in text:
            return age >= nums[0]
        return age == nums[0]
    return False

def calculate_quote(policy_id: str, age: int, requested_si: int, zone: str = "All") -> dict:
    # 1. Fetch a broad block of premiums for this policy to search in-memory
    url = f"{SUPABASE_URL}/rest/v1/policy_premiums?policy_id=eq.{policy_id}&limit=3000"
    all_premiums = requests.get(url, headers=_h()).json()
    
    if not all_premiums:
        return {"status": "error", "message": f"No premium rows found in DB for ID '{policy_id}'."}

    # 2. Find the correct Age Band dynamically
    matched_band = None
    for p in all_premiums:
        if matches_age(p['age_band'], age):
            matched_band = p['age_band']
            break
            
    if not matched_band:
        return {"status": "error", "message": f"Policy {policy_id} does not cover age {age}."}

    # 3. Filter premiums by this age band and zone
    age_filtered = [p for p in all_premiums if p['age_band'] == matched_band]
    
    # Try exact zone match first
    zone_filtered = [p for p in age_filtered if str(p.get('zone', '')).lower() == zone.lower()]
    
    # Fallback 1: Try 'All', '', or 'None'
    if not zone_filtered:
        zone_filtered = [p for p in age_filtered if str(p.get('zone', '')).lower() in ["all", "", "none"]]
        
    # Fallback 2: The Zone Trap! Grab whatever zone is available if 'All' isn't explicitly there
    zone_note = None
    if not zone_filtered and age_filtered:
        available_zones = sorted(list(set([str(p.get('zone', '')) for p in age_filtered])))
        if available_zones:
            fallback_zone = available_zones[0]
            zone_filtered = [p for p in age_filtered if str(p.get('zone', '')) == fallback_zone]
            zone_note = f"Zone adjusted to {fallback_zone}"

    if not zone_filtered:
        return {"status": "error", "message": f"No zone data found for age {age}."}

    # 4. Smart Sum Insured Matching (Find closest available)
    available_sis = sorted(list(set([p['sum_insured'] for p in zone_filtered])))
    if not available_sis:
        return {"status": "error", "message": f"No SI data found for age {age}."}

    target_si = requested_si
    si_note = None
    
    if requested_si not in available_sis:
        closest_si = min(available_sis, key=lambda x: abs(x - requested_si))
        target_si = closest_si
        si_note = f"Requested ₹{requested_si/100000:g}L unavailable. Quoted closest match: ₹{target_si/100000:g}L"

    # Retrieve the exact base premium for our finalized SI
    base_premium = next((p['base_premium'] for p in zone_filtered if p['sum_insured'] == target_si), 0)
    
    if not base_premium:
         return {"status": "error", "message": f"Premium rate missing for {target_si}."}

    # 5. Apply Discounts & Loadings
    disc_url = f"{SUPABASE_URL}/rest/v1/policy_discounts?policy_id=eq.{policy_id}"
    discounts = requests.get(disc_url, headers=_h()).json() or []
    
    load_url = f"{SUPABASE_URL}/rest/v1/policy_loadings?policy_id=eq.{policy_id}"
    loadings = requests.get(load_url, headers=_h()).json() or []

    net_premium = base_premium
    applied_modifiers = []
    
    for d in discounts:
        if any(k in d['discount_type'].lower() for k in ["online", "direct"]):
            deduction = base_premium * (float(d['discount_percentage']) / 100)
            net_premium -= deduction
            applied_modifiers.append(f"{d['condition_text']} (-{d['discount_percentage']}%)")
            
    gst = net_premium * 0.18
    final_premium = net_premium + gst
    
    return {
        "status": "success",
        "base_premium": int(base_premium),
        "applied_modifiers": applied_modifiers,
        "net_premium_before_tax": int(net_premium),
        "gst_amount": int(gst),
        "final_payable": int(final_premium),
        "actual_si_used": target_si,
        "si_note": si_note,
        "zone_note": zone_note,
        "all_discounts": discounts,
        "all_loadings": loadings
    }