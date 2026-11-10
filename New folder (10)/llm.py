
import hashlib
import json
import re
import time

import diskcache
import config

_cache = diskcache.Cache(config.CACHE_DIR)


def _messages(prompt, system):
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": prompt}]


def _gemini(prompt, system):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    cfg = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=0.2
    )

    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=cfg
    )

    return resp.text or ""


def _groq(prompt, system):
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)

    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=0.2,
        messages=_messages(prompt, system)
    )

    return resp.choices[0].message.content or ""


def _ollama(prompt, system):
    import ollama

    resp = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=_messages(prompt, system),
        options={"temperature": 0.2}
    )

    return resp["message"]["content"]


PROVIDERS = {
    "gemini": _gemini,
    "groq": _groq,
    "ollama": _ollama
}


def ask(prompt, system="", cache=True, provider=None):
    """Common interface for LLM calls, with caching and retries."""

    provider = provider or config.LLM_PROVIDER

    models = (
        f"{config.GEMINI_MODEL}|"
        f"{config.GROQ_MODEL}|"
        f"{config.OLLAMA_MODEL}"
    )

    key = hashlib.sha256(
        f"{provider}|{models}|{system}|{prompt}".encode()
    ).hexdigest()

    if cache and key in _cache:
        return _cache[key]

    last_error = None

    for attempt in range(4):
        try:
            text = PROVIDERS[provider](prompt, system)
            break
        except Exception as e:
            last_error = e
            if attempt < 3:
                time.sleep(2 ** attempt)
    else:
        raise RuntimeError(
            f"LLM call failed after 4 tries: {last_error}"
        )

    if cache:
        _cache[key] = text

    return text


def parse_json(text):
    """Extract the first JSON list or object from a model reply."""

    text = re.sub(r"```(?:json)?", "", text).strip()

    spots = [
        (text.find(o), o, c)
        for o, c in (("[", "]"), ("{", "}"))
        if text.find(o) != -1
    ]

    for start, _, closer in sorted(spots):
        end = text.rfind(closer)

        if end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue

    return None


def ask_json(prompt, system="", cache=True, provider=None):
    return parse_json(ask(prompt, system, cache, provider))


