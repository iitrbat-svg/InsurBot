"""
Pricing Calculation Module
Resolves matrix bands and matches dynamic rule modifications from Supabase.
"""
import os
import requests
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

def resolve_age_band(age: int) -> str:
    """
    Normalizes an integer age into structural CSV matching bands.
    Adjust these strings if your CSV files use different labels (e.g., '36 to 45').
    """
    if age <= 17: return "0-17"
    if age <= 25: return "18-25"
    if age <= 35: return "26-35"
    if age <= 45: return "36-45"
    if age <= 50: return "46-50"
    if age <= 55: return "51-55"
    if age <= 60: return "56-60"
    if age <= 65: return "61-65"
    if age <= 70: return "66-70"
    return ">70"

def calculate_quote(policy_id: str, age: int, sum_insured: int = 500000, zone: str = "All") -> dict:
    """Fetches exact baseline records and chains multi-conditional tabular rule evaluations."""
    age_band = resolve_age_band(age)
    
    # 1. Match standard matrix row
    url = f"{SUPABASE_URL}/rest/v1/policy_premiums?policy_id=eq.{policy_id}&age_band=eq.{age_band}&sum_insured=eq.{sum_insured}&limit=1"
    
    print(f"\n[DEBUG PRICING ENGINE] Fetching Primary URL: {url}")
    res = requests.get(url, headers=_h()).json()
    
    # Optional fallback search to check for general 'All' zone grids if strict match yields empty sets
    if not res and zone != "All":
        fallback_url = f"{SUPABASE_URL}/rest/v1/policy_premiums?policy_id=eq.{policy_id}&age_band=eq.{age_band}&sum_insured=eq.{sum_insured}&zone=eq.All&limit=1"
        print(f"[DEBUG PRICING ENGINE] Fetching Fallback URL: {fallback_url}")
        res = requests.get(fallback_url, headers=_h()).json()
        
    if not res:
        print(f"[DEBUG PRICING ENGINE] Failed! DB returned: {res}")
        return {
            "status": "error", 
            "message": f"Premium record unavailable for Age {age} (Band {age_band}) with SI {sum_insured}."
        }
        
    base_premium = res[0]['base_premium']
    
    # 2. Collect linked policy rule tables
    disc_url = f"{SUPABASE_URL}/rest/v1/policy_discounts?policy_id=eq.{policy_id}"
    discounts = requests.get(disc_url, headers=_h()).json() or []
    
    load_url = f"{SUPABASE_URL}/rest/v1/policy_loadings?policy_id=eq.{policy_id}"
    loadings = requests.get(load_url, headers=_h()).json() or []
    
    # 3. Apply operational math adjustments
    net_premium = base_premium
    applied_modifiers = []
    
    # Apply standard online discount deduction defaults if detected in metadata records
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
        "all_discounts": discounts,
        "all_loadings": loadings
    }