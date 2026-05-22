"""
STEP 6: RAG Retriever Node
"""
import os, re
from dotenv import load_dotenv
import requests

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TOP_K        = 8
MIN_SIM      = 0.3

CF_ACCOUNT_ID = os.environ["CF_ACCOUNT_ID"]
CF_AI_TOKEN   = os.environ["CF_AI_TOKEN"]
CF_EMBED_URL  = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/@cf/baai/bge-base-en-v1.5"

def _embed(text: str) -> list[float]:
    r = requests.post(
        CF_EMBED_URL,
        headers={"Authorization": f"Bearer {CF_AI_TOKEN}"},
        json={"text": [f"Represent this sentence: {text}"]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["result"]["data"][0]

def warmup():
    print("Embedder ready (Cloudflare AI).")

def _h():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }

def clean_chunk(text: str) -> str:
    text = re.sub(r'(\|[ \t]*){3,}', '| ', text)           # collapse pipe spam
    text = re.sub(r'^\s*[-|]{3,}\s*$', '', text, flags=re.MULTILINE)  # table dividers
    text = re.sub(r'[*_]{2,}', '', text)                    # bold/italic
    text = re.sub(r'#{1,4}\s*', '', text)                   # headings
    text = re.sub(r'\n{3,}', '\n\n', text)                  # blank lines
    text = re.sub(r'[ \t]{2,}', ' ', text)                  # spaces
    return text.strip()

def _enrich(chunks: list) -> list:
    ids = list({c["policy_id"] for c in chunks})
    r   = requests.get(
        f"{SUPABASE_URL}/rest/v1/policies?id=in.({','.join(ids)})&select=id,product_slug,insurer",
        headers=_h(), timeout=10)
    meta = {row["id"]: row for row in (r.json() if r.ok else [])}
    for c in chunks:
        m = meta.get(c["policy_id"], {})
        c["product_slug"] = m.get("product_slug", c["policy_id"])
        c["insurer"]      = m.get("insurer", "")
        c["chunk_text"]   = clean_chunk(c["chunk_text"])
    return chunks

def retrieve(
    query: str,
    policy_ids: list | None = None,
    section_tags: list | None = None,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIM,
) -> dict:
    embedding = _embed(query)
    payload   = {
        "query_embedding":     embedding,
        "filter_policy_ids":   policy_ids   or None,
        "filter_section_tags": section_tags or None,
        "match_count":         top_k,
        "min_similarity":      min_similarity,
    }
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_policy_chunks",
        headers=_h(), json=payload, timeout=15)

    chunks = r.json() if r.ok else []

    # Retry with lower threshold if nothing returned
    if not chunks:
        payload["min_similarity"] = 0.15
        r2     = requests.post(f"{SUPABASE_URL}/rest/v1/rpc/match_policy_chunks",
                               headers=_h(), json=payload, timeout=15)
        chunks = r2.json() if r2.ok else []

    if chunks:
        chunks = _enrich(chunks)

    return {
        "chunks":                chunks,
        "query":                 query,
        "policy_ids_searched":   policy_ids   or ["all"],
        "section_tags_searched": section_tags or ["all"],
    }

def format_chunks(result: dict, max_chunks: int = 6) -> str:
    chunks = result["chunks"][:max_chunks]
    if not chunks:
        return "No relevant policy text found."
    grouped: dict[str, list] = {}
    for c in chunks:
        grouped.setdefault(c.get("product_slug", c["policy_id"]), []).append(c)
    lines = []
    for policy_name, pchunks in grouped.items():
        lines.append(f"\n### {pchunks[0].get('insurer','')} — {policy_name}")
        for c in pchunks:
            lines.append(f"[{c.get('section_tag','')} | sim:{c.get('similarity',0):.2f}]")
            lines.append(c["chunk_text"])
    return "\n".join(lines)

# Warm up model at import time if running as server
def warmup():
    _embed("warmup")
    print("Embedder ready.")
