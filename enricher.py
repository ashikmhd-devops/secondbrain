import json

from ollama import ollama_generate

_PROMPT_TEMPLATE = """
You are a universal data extraction assistant. Analyze the input and extract ALL structured information.

Input: {raw_text}

Respond ONLY with strictly valid JSON:
{{
  "title": "Short descriptive title (max 80 chars)",
  "category": "Exactly one of: Person | Finance | Credentials | Event | Idea | Task | Health | Travel | Reference | General",
  "tags": ["relevant", "searchable", "tags"],
  "data_types": ["list ALL detected types from: text | number | currency | date | time | account_number | card_number | ifsc_code | upi_id | phone | email | url | api_key | password | token | otp | pan_number | aadhaar | passport | driving_license | vehicle_number | name | address | coordinates | measurement | percentage | ip_address | id_number"],
  "extracted_entities": {{
    "names": [], "amounts": [], "currencies": [], "dates": [],
    "phones": [], "emails": [], "urls": [],
    "account_numbers": [], "card_numbers": [], "ifsc_codes": [], "upi_ids": [],
    "pan_numbers": [], "api_keys_or_tokens": [], "passwords": [],
    "addresses": [], "id_numbers": [], "other": {{}}
  }},
  "is_sensitive": true or false
}}

Rules:
- is_sensitive = true if input contains passwords, API keys, tokens, OTPs, card/account numbers, PAN, Aadhaar
- Extract exact raw values, never paraphrase numbers or keys
- Output raw JSON only, no markdown
"""


def _fallback(raw_text: str) -> dict:
    return {
        "title": raw_text[:80],
        "category": "General",
        "tags": [],
        "data_types": ["text"],
        "extracted_entities": {},
        "is_sensitive": False,
    }


async def enrich_memory(raw_text: str) -> dict:
    prompt = _PROMPT_TEMPLATE.format(raw_text=raw_text)
    try:
        response = await ollama_generate(prompt)
        return json.loads(response)
    except Exception:
        return _fallback(raw_text)
