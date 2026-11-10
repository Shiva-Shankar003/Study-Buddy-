import base64
import datetime as dt
import hashlib
import json
import os
import re
import time

import pandas as pd
import streamlit as st

import config
import db
import learning
import planner
from agent import run
from ingest import ingest_image, ingest_pdf, ingest_youtube
from quiz import make_flashcards, make_quiz
from video_analyzer import (FORMATS, analyze_locally, analyze_with_gemini,
                            make_highlights, make_notes)

st.set_page_config(page_title="StudyBuddy", layout="wide")
st.title("StudyBuddy: Agentic Multimodal RAG")

st.markdown("""<style>
.st-key-quickbar {position: sticky; bottom: 5rem; z-index: 50; background: Canvas;
                  padding: 0.3rem 0;}
.st-key-quickbar button {min-height: 2rem; padding: 0.1rem 0.5rem; font-size: 0.85rem;}
</style>""", unsafe_allow_html=True)

NEW_SUBJECT = "➕ New subject..."
REMOVE_SUBJECT = "➖ Remove subject..."
WHOLE = "— whole subject (normal chat) —"
VIDEOS_MODE = "🎬 Explore videos"
EVERYTHING = "📚 Everything uploaded in this subject"
QUICK_LABELS = ["Highlights", "Short summary", "Detailed notes", "Mind map", "Flowchart"]
NOTE_CHOICES = ["Short summary", "Detailed notes", "Bullet points only", "Exam revision sheet",
                "Step-by-step explanation", "Table of key terms", "Questions and answers"]
IMAGE_WORDS = ("image", "images", "photo", "photos", "picture", "pictures",
               "इमेज", "फोटो", "फ़ोटो", "तस्वीर", "चित्र")
MINDMAP_WORDS = ("mind map", "mindmap", "mind-map", "माइंड मैप", "माइंडमैप", "माइंड-मैप")
FLOW_WORDS = ("flow chart", "flowchart", "flow-chart", "फ्लो चार्ट", "फ्लोचार्ट", "फ्लो-चार्ट")
NOTE_STRONG = ("summar", "highlight", "revision", "cheat sheet", "cheatsheet", "bullet",
               "outline", "सारांश", "संक्षेप", "हाइलाइट", "रिवीजन", "समरी", "समराइज")
NOTE_VERBS = ("make", "create", "generate", "write", "give", "prepare", "banao", "bana",
              "बना", "दो", "चाहिए", "लिख", "तैयार")
FORMAT_HINTS = ("table", "q&a", "qna", "step-by-step", "step by step", "detailed", "short",
                "टेबल", "विस्तृत")


def save_upload(f):
    path = os.path.join(config.UPLOAD_DIR, f.name)
    with open(path, "wb") as out:
        out.write(f.getbuffer())
    return path


def drop_file(name, subject):
    if db.name_used_elsewhere(name, subject):
        return
    path = os.path.join(config.UPLOAD_DIR, name)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def remove_doc(subject, doc, kind):
    db.delete_doc(subject, doc)
    if kind != "youtube":
        drop_file(doc, subject)


def remove_video(subject, name):
    db.delete_video(subject, name)
    drop_file(name, subject)
    for store in ("video_highlights", "video_notes"):
        st.session_state.get(store, {}).pop(f"{subject}::{name}", None)


def remove_subject(subject):
    gone = db.delete_subject(subject)
    for d in gone["docs"]:
        if d["kind"] != "youtube":
            drop_file(d["doc"], subject)
    for name in gone["videos"]:
        drop_file(name, subject)
    st.session_state.get("chats", {}).pop(subject, None)
    st.session_state.get("pending", {}).pop(subject, None)
    for store in ("video_highlights", "video_notes"):
        d = st.session_state.get(store, {})
        for k in [k for k in d if k.startswith(f"{subject}::")]:
            d.pop(k, None)


def llm_chat(system, messages):
    """One chat call. messages = [{'role': 'user'|'assistant', 'content': str}, ...]"""
    import requests
    provider = os.getenv("SB_LLM_PROVIDER", "groq").lower()
    hist = list(messages)
    while hist and hist[0]["role"] != "user":
        hist.pop(0)
    if provider == "gemini":
        model = os.getenv("SB_GEMINI_MODEL", "gemini-flash-latest")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        contents = [{"role": "user" if m["role"] == "user" else "model",
                     "parts": [{"text": m["content"]}]} for m in hist]
        r = requests.post(url, params={"key": os.getenv("GEMINI_API_KEY")},
                          json={"system_instruction": {"parts": [{"text": system}]},
                                "contents": contents}, timeout=120)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    model = os.getenv("SB_GROQ_MODEL", "openai/gpt-oss-120b")
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                      json={"model": model,
                            "messages": [{"role": "system", "content": system}] + hist},
                      timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def fit_text(pieces, limit=12000):
    """Join text pieces; if too long, sample evenly across all of them."""
    pieces = [p for p in pieces if p]
    if sum(len(p) for p in pieces) <= limit:
        return "\n\n".join(pieces)
    n = len(pieces)
    k = min(n, max(1, limit // 800))
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)}) if k > 1 else [0]
    per = limit // len(idx)
    return "\n\n".join(pieces[i][:per] for i in idx)


def split_text(t, size=1500):
    return [t[i:i + size] for i in range(0, len(t), size)]


def material_text(active, src, subj_videos, limit=12000):
    if src.startswith("🎬 "):
        return fit_text(split_text(subj_videos[src[2:].strip()]), limit)
    if src.startswith("📄 "):
        doc = src[2:].strip()
        return fit_text([c["text"] for c in db.all_chunks(active) if c["doc"] == doc], limit)
    pieces = [c["text"] for c in db.all_chunks(active)]
    for a in subj_videos.values():
        pieces += split_text(a)
    return fit_text(pieces, limit)


def lang_line(lang):
    return {"English": "Write in English.", "Hindi": "Write in Hindi."}.get(
        lang, "Write in the same language as the user's request.")


def ask_source(question, text, src, lang="Same as the video", history=None):
    kind = "video" if src.startswith("🎬 ") else "document"
    extra = " Mention timestamps when useful." if kind == "video" else ""
    system = (f"You are a helpful study assistant. The user is studying one {kind}; its text "
              "is below. For questions about what it says, use the text." + extra
              + " For general or concept questions (for example 'what is X'), answer from "
              "your own knowledge, explain it well, and connect it to the material when "
              "that helps. When part of the answer comes from your own knowledge rather "
              "than the material, say so briefly. Never invent details about the material "
              "that are not in the text. " + lang_line(lang) + "\n\nMATERIAL:\n" + text)
    hist = [{"role": m["role"], "content": m["content"][:1500]} for m in (history or [])[-6:]]
    return llm_chat(system, hist + [{"role": "user", "content": question}])


def study_notes(text, request, lang):
    system = ("You write study notes. Using ONLY the material below, do exactly what the user "
              "asks, in the format the user asks for (for example summary, bullet points, "
              "table, exam revision sheet, step-by-step, Q&A, cheat sheet). Do not add facts "
              "that are not in the material. Use clean Markdown. " + lang_line(lang)
              + "\n\nMATERIAL:\n" + text)
    return llm_chat(system, [{"role": "user", "content": request}])


def topic_text(topic):
    """Write a study explanation of a topic from the model's own knowledge."""
    system = ("You are a knowledgeable teacher. Write a clear, accurate, well-structured "
              "explanation of the topic the user gives, covering the key ideas, definitions, "
              "parts, processes and examples, in about 600 words.")
    return llm_chat(system, [{"role": "user", "content": topic}])


def _clean(t, n=60):
    t = re.sub(r'["\[\]\(\)\{\}<>|;#`\\]', " ", str(t))
    t = " ".join(t.split())
    return t[:n] or "-"


def _name(x):
    return x if isinstance(x, str) else (x.get("name") or x.get("text") or "") if isinstance(x, dict) else str(x)


def build_mermaid(kind, data):
    if kind == "mindmap":
        branches = data.get("branches", [])[:8]
        if not branches:
            raise ValueError("the model returned no branches")
        lines = ["graph LR", f'  r["{_clean(data.get("title", "Mind map"), 50)}"]']
        for i, b in enumerate(branches, 1):
            lines.append(f'  r --> b{i}["{_clean(_name(b), 40)}"]')
            kids = b.get("children", []) if isinstance(b, dict) else []
            for j, c in enumerate(kids[:6], 1):
                lines.append(f'  b{i} --> c{i}_{j}["{_clean(_name(c), 50)}"]')
        lines += ["  classDef root fill:#4f46e5,color:#ffffff,stroke:none",
                  "  classDef br fill:#e0e7ff,stroke:#4f46e5",
                  "  class r root",
                  "  class " + ",".join(f"b{i}" for i in range(1, len(branches) + 1)) + " br"]
        return "\n".join(lines)
    nodes = data.get("nodes", [])[:16]
    if len(nodes) < 2:
        raise ValueError("the model returned too few steps")
    ids, lines = {}, ["flowchart TD"]
    for i, n in enumerate(nodes, 1):
        nid = f"n{i}"
        ids[str(n.get("id", nid))] = nid
        text = _clean(n.get("text", ""), 60)
        t = str(n.get("type", "step")).lower()
        if t in ("start", "end"):
            shape = f'(["{text}"])'
        elif t == "decision":
            shape = '{"' + text + '"}'
        else:
            shape = f'["{text}"]'
        lines.append(f"  {nid}{shape}")
    for e in data.get("edges", []):
        a, b = ids.get(str(e.get("from"))), ids.get(str(e.get("to")))
        if not a or not b:
            continue
        lab = _clean(e["label"], 20) if e.get("label") else ""
        lines.append(f"  {a} -->|{lab}| {b}" if lab else f"  {a} --> {b}")
    return "\n".join(lines)


def diagram_code(kind, text, request, lang):
    if kind == "mindmap":
        shape = ('Return ONLY JSON: {"title": "...", "branches": [{"name": "...", '
                 '"children": ["...", "..."]}]}. Use 4-7 branches, each with 2-5 short '
                 'children (max 6 words each).')
    else:
        shape = ('Return ONLY JSON: {"title": "...", "nodes": [{"id": "n1", "text": "...", '
                 '"type": "start|step|decision|end"}], "edges": [{"from": "n1", "to": "n2", '
                 '"label": ""}]}. Use 6-14 nodes with short text (max 7 words). Decision '
                 'nodes need two outgoing edges labeled like Yes / No.')
    lang_rule = {"English": "All text in English.", "Hindi": "All text in Hindi."}.get(
        lang, "All text in the same language as the user's request.")
    system = ("You turn study material into diagram data. Use ONLY the material below. "
              + shape + " " + lang_rule + " No explanation, no code fences.\n\nMATERIAL:\n"
              + text)
    raw = llm_chat(system, [{"role": "user", "content": request}])
    s, e = raw.find("{"), raw.rfind("}")
    return build_mermaid(kind, json.loads(raw[s:e + 1]))


def render_mermaid(code):
    import requests
    b64 = base64.urlsafe_b64encode(code.encode("utf-8")).decode()
    try:
        r = requests.get(f"https://mermaid.ink/img/{b64}",
                         params={"type": "png", "bgColor": "FFFFFF"}, timeout=60)
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            return r.content
    except Exception:
        pass
    r = requests.post("https://kroki.io/mermaid/png", data=code.encode("utf-8"),
                      headers={"Content-Type": "text/plain"}, timeout=60)
    r.raise_for_status()
    return r.content


def detect_intent(q):
    ql = q.lower()
    if any(w in ql for w in MINDMAP_WORDS):
        return "mindmap"
    if any(w in ql for w in FLOW_WORDS):
        return "flowchart"
    if any(w in ql for w in IMAGE_WORDS):
        return "images"
    if "highlight" in ql or "हाइलाइट" in ql:
        return "highlights"
    if any(w in ql for w in NOTE_STRONG):
        return "notes"
    if ("note" in ql or "नोट" in ql) and any(v in ql for v in NOTE_VERBS):
        return "notes"
    return "ask"


def format_given(q):
    ql = q.lower()
    return any(w in ql for w in NOTE_STRONG + FORMAT_HINTS)


def generate_reply(intent, request, src, lang, active, subj_videos, style=None, text=None):
    """Build a notes / highlights / diagram reply for the chosen source or given text."""
    is_video = text is None and src.startswith("🎬 ")
    if intent in ("mindmap", "flowchart"):
        if text is None:
            text = material_text(active, src, subj_videos)
        code = diagram_code(intent, text, request, lang)
        png = render_mermaid(code)
        what = "mind map" if intent == "mindmap" else "flowchart"
        return {"content": f"Here is your {what}:", "png": png, "code": code,
                "kind": "diagram"}
    if is_video:
        analysis = subj_videos[src[2:].strip()]
        if intent == "highlights":
            return {"content": make_highlights(analysis), "kind": "notes"}
        if style:
            return {"content": make_notes(analysis, style, lang), "kind": "notes"}
    if text is None:
        text = material_text(active, src, subj_videos)
    return {"content": study_notes(text, request, lang), "kind": "notes"}


def quick_format(label):
    """Quick-button label -> the matching key in FORMATS (None for non-note buttons)."""
    prefix = {"Short summary": "short summary", "Detailed notes": "detailed notes"}.get(label)
    if not prefix:
        return label
    return next((str(f) for f in FORMATS if str(f).lower().startswith(prefix)), label)


def style_request(style):
    """Turn a format label into a plain request for non-video material."""
    if style == "Highlights":
        return "Give the key highlights: only the most important points, as short bullets."
    if isinstance(FORMATS, dict) and FORMATS.get(style):
        return f"{style}. {FORMATS[style]}"
    return f"Make notes in this format: {style}"


def transcribe(audio_bytes):
    import requests
    data = {"model": os.getenv("SB_STT_MODEL", "whisper-large-v3-turbo"),
            "prompt": "A student's study question in Hindi (Devanagari) or English."}
    if os.getenv("SB_STT_LANG"):
        data["language"] = os.getenv("SB_STT_LANG")
    r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
                      headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                      files={"file": ("voice.wav", audio_bytes, "audio/wav")},
                      data=data, timeout=120)
    r.raise_for_status()
    return r.json().get("text", "").strip()


def _ddgs():
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    return DDGS()


def web_search(query, n=5):
    return list(_ddgs().text(query, max_results=n))


def web_images(query, n=3, errs=None):
    """Images from DuckDuckGo; if that fails, from Wikimedia Commons."""
    errs = errs if errs is not None else []
    urls = []
    try:
        res = list(_ddgs().images(query, max_results=n + 6))
        urls = [r.get("thumbnail") or r.get("image") for r in res]
        urls = [u for u in urls if u]
        if not urls:
            errs.append("DuckDuckGo returned no images")
    except Exception as e:
        errs.append(f"DuckDuckGo: {e}")
    if len(urls) < n:
        try:
            import requests
            r = requests.get("https://commons.wikimedia.org/w/api.php",
                             params={"action": "query", "generator": "search",
                                     "gsrnamespace": 6, "gsrsearch": query, "gsrlimit": 12,
                                     "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 500,
                                     "format": "json"},
                             headers={"User-Agent": "StudyBuddy/1.0 (student project)"},
                             timeout=20)
            pages = (r.json().get("query") or {}).get("pages", {})
            for pg in pages.values():
                info = (pg.get("imageinfo") or [{}])[0]
                u = info.get("thumburl") or info.get("url")
                if u and u.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png")):
                    urls.append(u)
        except Exception as e:
            errs.append(f"Wikimedia: {e}")
    return urls[:n]


_IMG_FILLER = re.compile(
    r"\b(show|get|give|find|display|some|me|of|the|a|an|about|please|and|with|for|more|"
    r"ki|ka|ke|dikhao|dikha|dikhaiye|do|dijiye|batao|chahiye|kuch|mujhe|"
    r"images?|photos?|pictures?|pics?)\b|इमेज(ेस)?|फोटो|फ़ोटो|तस्वीर(ें)?|चित्र|दिखाओ|दिखाइए|"
    r"दिखा|दो|दीजिए|चाहिए|का|की|के|कुछ|मुझे|ढूंढो",
    re.IGNORECASE)


def image_topic(text):
    return " ".join(_IMG_FILLER.sub(" ", text).split())


def last_topic(messages):
    """The most recent topic the user asked about (a question or an images request)."""
    for m in reversed(messages):
        if m["role"] != "user" or m.get("kind") in ("tool", "web"):
            continue
        it = detect_intent(m["content"])
        if it == "images":
            prev = image_topic(m["content"])
            if len(prev) >= 3:
                return prev
        elif it == "ask":
            return m["content"]
    return ""


def image_query(question, messages, active):
    """Topic to search images for: the request without filler, else the last topic."""
    q = image_topic(question)
    if len(q) >= 3:
        return q
    return last_topic(messages[:-1]) or active or question


def material_options(active, subj_videos):
    opts = {EVERYTHING: WHOLE}
    for d in db.list_docs(active):
        opts[f"📄 {d['doc']}"] = f"📄 {d['doc']}"
    for n in subj_videos:
        opts[f"🎬 {n}"] = f"🎬 {n}"
    return opts


def run_guided(pend, use_material, src, topic, fmt, lang, active, subj_videos):
    """Create notes / highlights / a diagram for what the user picked."""
    intent = pend["intent"]
    fmt = fmt or pend.get("fmt")
    if intent == "highlights":
        request = style_request("Highlights")
    elif intent == "notes" and fmt:
        request = style_request(fmt)
    elif pend.get("request"):
        request = pend["request"]
    else:
        request = "Create a mind map" if intent == "mindmap" else "Create a flowchart"
    style = None
    if intent == "notes" and use_material and src.startswith("🎬 "):
        if fmt:
            key = quick_format(fmt)
            style = next((f for f in FORMATS if str(f) == key), None)
        else:
            style = next((f for f in FORMATS if str(f).lower() in request.lower()), None)
    if use_material:
        return generate_reply(intent, request, src, lang, active, subj_videos, style=style)
    text = topic_text(topic)
    return generate_reply(intent, f"{request}\nTopic: {topic}", WHOLE, lang, active,
                          subj_videos, text=text)


def web_more(question):
    """Short web-based explanation + links + 2 images."""
    results = web_search(question, 5)
    imgs = web_images(question, 2)
    snippets = "\n".join(f"[{i}] {r.get('title', '')}: {r.get('body', '')}"
                         for i, r in enumerate(results, 1))
    system = ("You are a study assistant. Using ONLY the web search results below, add useful "
              "extra information to the answer already given. Keep it short and accurate, "
              "cite sources like [1], and reply in the same language as the question.\n\n"
              "WEB RESULTS:\n" + snippets)
    text = llm_chat(system, [{"role": "user", "content": question}])
    links = "\n\n".join(f"[{i}] [{r.get('title', 'link')}]({r.get('href', '')})"
                        for i, r in enumerate(results, 1))
    return text, links, imgs


subjects = db.list_subjects()

nxt = st.session_state.pop("_next_subject", None)
if nxt and (nxt in subjects or nxt == NEW_SUBJECT):
    st.session_state["sb_subject_pick"] = nxt

with st.sidebar:
    st.header("Add study material")

    if st.session_state.get("flash"):
        st.info(st.session_state.pop("flash"))

    if subjects:
        pick = st.selectbox("Subject", subjects + [NEW_SUBJECT, REMOVE_SUBJECT],
                            key="sb_subject_pick")
    else:
        pick = NEW_SUBJECT
    removing = pick == REMOVE_SUBJECT
    if pick == NEW_SUBJECT:
        subject = st.text_input("New subject name", placeholder="e.g. Maths, DBMS").strip()
    elif removing:
        subject = ""
    else:
        subject = pick

    active = pick if (pick not in (NEW_SUBJECT, REMOVE_SUBJECT) and subjects) else None

    if removing:
        target = st.selectbox("Subject to remove", subjects, key="rm_target")
        st.warning(f"This will permanently delete {len(db.list_docs(target))} files, "
                   f"{len(db.list_videos(target))} videos and "
                   f"{len(db.list_generated(target))} generated items "
                   f"(notes, mind maps etc.) of '{target}'. This cannot be undone.")
        sure = st.checkbox("Yes, delete all data of this subject", key=f"rm_ok_{target}")
        if st.button("🗑 Delete subject permanently", disabled=not sure, type="primary"):
            remove_subject(target)
            rest = [x for x in subjects if x != target]
            st.session_state.pop(f"rm_ok_{target}", None)
            st.session_state["flash"] = f"'{target}' and all its data were deleted."
            st.session_state["_next_subject"] = rest[0] if rest else NEW_SUBJECT
            st.rerun()
        pdfs, images, videos, vmode, video = [], [], [], "", ""
    else:
        pdfs = st.file_uploader("PDF notes", type="pdf", accept_multiple_files=True)
        images = st.file_uploader("Handwritten pages", type=["png", "jpg", "jpeg"],
                                  accept_multiple_files=True)
        videos = st.file_uploader("Videos (your own video files)",
                                  type=["mp4", "mov", "mkv", "webm", "avi", "m4v"],
                                  accept_multiple_files=True)
        vmode = st.radio(
            "Video analysis",
            ["Full: sees pictures + hears audio (Gemini)",
             "Local: speech + on-screen text (free)"],
            key="vid_mode")
        video = st.text_input("YouTube link (optional)")

    if not removing and st.button("Add to knowledge base"):
        if not subject:
            st.warning("Type a subject name first.")
        else:
            added, errors, done_videos = 0, [], []
            with st.spinner("Reading and indexing..."):
                for f in pdfs:
                    added += ingest_pdf(save_upload(f), subject)
                for f in images:
                    added += ingest_image(save_upload(f), subject)
                if video.strip():
                    try:
                        added += ingest_youtube(video.strip(), subject)
                    except Exception as e:
                        errors.append(f"YouTube failed: {e}")
                for vf in videos:
                    try:
                        vpath = save_upload(vf)
                        if vmode.startswith("Full"):
                            analysis = analyze_with_gemini(vpath)
                        else:
                            analysis = analyze_locally(vpath)
                        db.save_video(subject, vf.name, analysis)
                        done_videos.append(vf.name)
                    except Exception as e:
                        errors.append(f"Video '{vf.name}' failed: {e}")
            msg = f"Added {added} chunks to '{subject}'."
            if added == 0 and not errors and not done_videos:
                msg += " (0 usually means this file or link was already added earlier.)"
            if done_videos:
                msg += ("\n\nVideo analyzed: " + ", ".join(done_videos)
                        + ". Open the Materials tab, or ask the chatbot for highlights/notes.")
            if errors:
                msg += "\n\n" + "\n\n".join(errors)
            st.session_state["flash"] = msg
            st.session_state["_next_subject"] = subject
            st.rerun()

    st.divider()
    fast = st.toggle("Fast mode (skip extra checks)", value=True)

section = st.radio("Section", ["Chat", "Quiz", "Review cards", "Progress",
                               "Study plan", "Materials"],
                   horizontal=True, label_visibility="collapsed", key="nav_section")

if section == "Chat":
    if not active:
        st.info("Pick a subject from the dropdown in the sidebar. "
                "If you just created a new one, add some material and press "
                "'Add to knowledge base'; it will be selected automatically.")
    else:
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**Subject: {active}**")
        search_all = c1.checkbox("Search all subjects instead", key="chat_all")
        scope = None if search_all else active
        flags = dict(use_verify=False, use_crag=False) if fast else {}

        chats = st.session_state.setdefault("chats", {})
        pending = st.session_state.setdefault("pending", {})
        if c2.button("🆕 New chat", key="new_chat_btn"):
            chats[active] = []
            pending.pop(active, None)
            st.rerun()
        messages = chats.setdefault(active, [])

        subj_videos = {v["name"]: v["analysis"] for v in db.list_videos(active)}
        has_docs = bool(db.list_docs(active))
        modes = [WHOLE] + ([VIDEOS_MODE] if subj_videos else [])
        w1, w2, w3 = st.columns([2, 2, 1])
        mode = w1.selectbox("Work with", modes, key=f"chat_mode_{active}")
        src = WHOLE
        if mode == VIDEOS_MODE:
            vsel = w2.selectbox("Video", list(subj_videos), key=f"chat_vid_{active}")
            src = f"🎬 {vsel}"
        vlang = w3.selectbox("Notes language", ["Same as the video", "English", "Hindi"],
                             format_func=lambda x: "Same as source" if x.startswith("Same") else x,
                             key=f"chat_vlang_{active}")

        last_idx = len(messages) - 1
        for i, m in enumerate(messages):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("png"):
                    st.image(m["png"])
                    st.download_button("⬇ Download image", m["png"],
                                       file_name="diagram.png", mime="image/png",
                                       key=f"dlimg_{active}_{i}")
                    with st.expander("Diagram code"):
                        st.code(m.get("code", ""), language="text")
                if m.get("images"):
                    st.image(m["images"], width=240)
                if m.get("saved"):
                    st.caption("💾 Saved in Materials → Generated")
                if m.get("kind") == "notes":
                    st.download_button("⬇ Download notes (.md)", m["content"],
                                       file_name="notes.md", key=f"dlmd_{active}_{i}")
                if m.get("sources"):
                    with st.expander("Sources and checks"):
                        st.markdown(m["sources"])
                if (m["role"] == "assistant" and i == last_idx and i > 0
                        and m.get("kind") not in ("tool", "web", "diagram", "notes")):
                    if st.button("🌐 Search more on web", key=f"web_{active}_{i}"):
                        if m.get("kind") == "images":
                            topic = m.get("topic") or messages[i - 1]["content"]
                            shown = {u for mm in messages for u in (mm.get("images") or [])}
                            errs = []
                            more = [u for u in web_images(topic, 12, errs) if u not in shown][:4]
                            messages.append({"role": "user",
                                             "content": f"🌐 More images: {topic}", "kind": "web"})
                            messages.append({"role": "assistant", "kind": "images",
                                             "topic": topic, "images": more,
                                             "content": (f"More images for: {topic}" if more else
                                                         "No new images found.\n\n" + "\n".join(
                                                             f"- {e}" for e in errs))})
                            st.rerun()
                        q = messages[i - 1]["content"]
                        try:
                            with st.spinner("Searching the web..."):
                                text, links, imgs = web_more(q)
                        except Exception as e:
                            text, links, imgs = (
                                f"Web search failed: {e}\n\n"
                                "(If it says 'No module named ddgs', run: pip install ddgs)",
                                "", [])
                        messages.append({"role": "user", "content": f"🌐 More on the web: {q}",
                                         "kind": "web"})
                        messages.append({"role": "assistant", "content": text,
                                         "sources": links, "images": imgs, "kind": "web"})
                        st.rerun()

        panel_slot = st.container()

        clicked = None
        if has_docs or subj_videos:
            with st.container(key="quickbar"):
                for col, lab in zip(st.columns(len(QUICK_LABELS)), QUICK_LABELS):
                    if col.button(lab, key=f"qbtn_{active}_{lab}", use_container_width=True):
                        clicked = lab
        if clicked:
            q_intent = {"Highlights": "highlights", "Mind map": "mindmap",
                        "Flowchart": "flowchart"}.get(clicked, "notes")
            pending[active] = {"id": time.time_ns(), "intent": q_intent, "request": None,
                               "fmt": clicked if q_intent == "notes" else None,
                               "ask_format": False, "title": clicked}

        pend = pending.get(active)
        if pend:
            with panel_slot:
                with st.chat_message("assistant"):
                    pid = pend["id"]
                    is_img = pend["intent"] == "images"
                    st.markdown(f"**{pend['title']}**: what should I base it on?")
                    use_material = False
                    if not is_img:
                        has_any = has_docs or bool(subj_videos)
                        choice = st.radio("Use", ["Uploaded material", "Topic from chat"],
                                          index=0 if has_any else 1, horizontal=True,
                                          key=f"g_choice_{pid}")
                        use_material = choice == "Uploaded material"
                    picked, topic, fmt = WHOLE, "", None
                    if use_material:
                        opts = material_options(active, subj_videos)
                        label = st.selectbox("Which material?", list(opts), key=f"g_sel_{pid}")
                        picked = opts[label]
                    else:
                        topic = st.text_input("Topic", value=last_topic(messages) or active,
                                              key=f"g_topic_{pid}")
                    if pend["ask_format"]:
                        fmt = st.selectbox("What type of notes do you want?", NOTE_CHOICES,
                                           key=f"g_fmt_{pid}")
                    b1, b2 = st.columns(2)
                    go = b1.button("✅ Create", key=f"g_go_{pid}", use_container_width=True)
                    stop = b2.button("Cancel", key=f"g_stop_{pid}", use_container_width=True)
                    if stop:
                        pending.pop(active, None)
                        st.rerun()
                    if go:
                        if not use_material and not topic.strip():
                            st.warning("Please type a topic.")
                        else:
                            if is_img:
                                tp = topic.strip()
                                errs = []
                                with st.spinner("Finding images..."):
                                    imgs = web_images(tp, 4, errs)
                                messages.append({"role": "user",
                                                 "content": f"Show images of {tp}"})
                                messages.append({
                                    "role": "assistant", "kind": "images", "topic": tp,
                                    "images": imgs, "sources": "",
                                    "content": (f"Images for: {tp}" if imgs else
                                                f"I could not find images for '{tp}'.\n\n"
                                                + "\n".join(f"- {e}" for e in errs))})
                            else:
                                what = label if use_material else f"topic: {topic.strip()}"
                                messages.append({"role": "user", "kind": "tool",
                                                 "content": f"{pend['title']} — {what}"})
                                try:
                                    with st.spinner("Creating..."):
                                        reply = run_guided(pend, use_material, picked,
                                                           topic.strip(), fmt, vlang, active,
                                                           subj_videos)
                                        if reply.get("kind") in ("notes", "diagram"):
                                            db.save_generated(
                                                active, f"{pend['title']} — {what}",
                                                reply["kind"], reply["content"],
                                                reply.get("png"), reply.get("code"), what)
                                            reply["saved"] = True
                                except Exception as e:
                                    reply = {"content": f"Could not create this: {e}",
                                             "kind": "tool"}
                                reply.setdefault("sources", "")
                                reply["role"] = "assistant"
                                messages.append(reply)
                            pending.pop(active, None)
                            st.rerun()

        question, audio = None, None
        try:
            val = st.chat_input("Ask a question, or ask for notes, a mind map, images...",
                                accept_audio=True, audio_sample_rate=16000)
        except TypeError:
            val = st.chat_input("Ask a question, or ask for notes, a mind map, images...")
            if hasattr(st, "audio_input"):
                with st.popover("🎤"):
                    rec = st.audio_input("Record", key=f"voice_{active}")
                    if rec is not None:
                        d = rec.getvalue()
                        h = hashlib.md5(d).hexdigest()
                        if st.session_state.get("_last_voice") != h:
                            st.session_state["_last_voice"] = h
                            audio = rec
        if isinstance(val, str):
            question = val
        elif val is not None:
            question = getattr(val, "text", None) or None
            audio = getattr(val, "audio", None) or audio
        if audio is not None and not question:
            try:
                with st.spinner("Listening..."):
                    question = transcribe(audio.getvalue()) or None
                if not question:
                    st.warning("I could not hear anything. Please try again.")
            except Exception as e:
                st.error(f"Could not understand the audio: {e}")
        if question:
            pending.pop(active, None)
            messages.append({"role": "user", "content": question})
            intent = detect_intent(question)
            bare_images = intent == "images" and len(image_topic(question)) < 3
            if intent in ("mindmap", "flowchart", "highlights", "notes") or bare_images:
                titles = {"mindmap": "Mind map", "flowchart": "Flowchart",
                          "highlights": "Highlights", "notes": "Notes", "images": "Images"}
                pending[active] = {"id": time.time_ns(), "intent": intent,
                                   "request": question, "fmt": None,
                                   "ask_format": intent == "notes" and not format_given(question),
                                   "title": titles[intent]}
                messages.append({"role": "assistant", "kind": "tool", "sources": "",
                                 "content": "Choose below what to base it on 👇"})
                st.rerun()
            with st.chat_message("user"):
                st.markdown(question)
            reply = {"content": "", "sources": "", "kind": None}
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        if intent == "images":
                            errs = []
                            topic = image_query(question, messages, active)
                            imgs = web_images(topic, 4, errs)
                            if imgs:
                                content = f"Images for: {topic}"
                            else:
                                content = (f"I could not find images for '{topic}'.\n\n"
                                           + "\n".join(f"- {e}" for e in errs))
                            reply = {"content": content, "images": imgs, "kind": "images", "topic": topic}
                        elif src != WHOLE:
                            text = material_text(active, src, subj_videos)
                            reply = {"content": ask_source(question, text, src, vlang,
                                                           messages[:-1]), "kind": None}
                        else:
                            out = run(question, scope, **flags)
                            if out["intent"] == "quiz":
                                reply = {"content": "Open the **Quiz** tab to start a quiz.",
                                         "kind": None}
                            else:
                                lines = [f"[{i}] {c.get('ref', c.get('source', ''))}"
                                         for i, c in enumerate(out.get("chunks", []), 1)]
                                v = out.get("verification") or {}
                                if v.get("faithfulness") is not None:
                                    lines.append("Claims supported by the sources: "
                                                 f"{v['faithfulness']:.0%}")
                                reply = {"content": out.get("answer", ""),
                                         "sources": "\n\n".join(lines), "kind": None}
                    except Exception as e:
                        reply = {"content": f"Could not complete this: {e}", "kind": "tool"}
            reply["role"] = "assistant"
            reply.setdefault("sources", "")
            messages.append(reply)
            st.rerun()

if section == "Quiz":
    if not subjects:
        st.info("Add study material first.")
    else:
        qsub = st.selectbox("Subject", subjects, key="quiz_subject")
        n = st.slider("Number of questions", 3, 10, 5)
        level = st.selectbox("Difficulty", ["auto", "remember", "understand", "apply", "analyze"])
        if st.button("Generate quiz"):
            with st.spinner("Writing questions..."):
                st.session_state.quiz = make_quiz(qsub, n, None if level == "auto" else level)
                st.session_state.quiz_id = time.time()
        quiz = st.session_state.get("quiz", [])
        qid = st.session_state.get("quiz_id", 0)
        for i, item in enumerate(quiz):
            st.markdown(f"**Q{i + 1}. {item['question']}**")
            st.radio("Answer", item["options"], index=None, key=f"ans_{qid}_{i}",
                     label_visibility="collapsed")
        if quiz and st.button("Submit answers"):
            score = 0
            for i, item in enumerate(quiz):
                right = item["options"][item["answer_index"]]
                correct = st.session_state.get(f"ans_{qid}_{i}") == right
                score += int(correct)
                learning.record_attempt(item["subject"], item["topic"], correct)
                if not correct:
                    learning.add_card(item["subject"], item["topic"], item["question"],
                                      f"{right}. {item['explanation']}")
                st.write(f"Q{i + 1}: " + ("Correct. " if correct else "Not quite. ")
                         + f"{item['explanation']} (source: {item['ref']})")
            st.success(f"Score: {score}/{len(quiz)}")

if section == "Review cards":
    if subjects:
        fsub = st.selectbox("Make flashcards for", subjects, key="flash_subject")
        if st.button("Generate flashcards from my notes"):
            with st.spinner("Writing flashcards..."):
                st.success(f"Created {make_flashcards(fsub)} cards")
    due = learning.due_cards(limit=1)
    if not due:
        st.info("No cards are due. Take a quiz or generate flashcards.")
    else:
        card = due[0]
        st.markdown(f"**{card['subject']} / {card['topic']}**")
        st.markdown(card["front"])
        if st.checkbox("Show answer", key=f"show_{card['id']}"):
            st.markdown(card["back"])
        for col, (label, q) in zip(st.columns(4),
                                   [("Again", 1), ("Hard", 3), ("Good", 4), ("Easy", 5)]):
            if col.button(label, key=f"rate_{card['id']}_{q}"):
                learning.review_card(card["id"], q)
                st.rerun()

if section == "Progress":
    for s in subjects:
        scores = learning.mastery(s)
        st.subheader(s)
        if not scores:
            st.caption("No quiz data yet.")
            continue
        st.bar_chart(pd.DataFrame({"mastery": scores}))
        weak = learning.weakest_topics(s)
        if weak:
            st.write("Weak topics:", ", ".join(weak))

if section == "Study plan":
    chosen = st.multiselect("Subjects", subjects, default=subjects)
    exam = st.date_input("Exam date", dt.date.today() + dt.timedelta(days=14))
    hours = st.slider("Study hours per day", 1.0, 6.0, 2.0, 0.5)
    if st.button("Make plan") and chosen:
        st.dataframe(pd.DataFrame(planner.make_plan(chosen, exam, hours)))

if section == "Materials":
    if not active:
        st.info("Pick a subject in the sidebar and all its material will show up here.")
    else:
        st.subheader(f"Materials: {active}")
        docs = db.list_docs(active)
        vids = db.list_videos(active)
        gens = db.list_generated(active)
        tab_up, tab_gen = st.tabs([f"📁 Uploaded material ({len(docs) + len(vids)})",
                                   f"✨ Generated ({len(gens)})"])

        with tab_up:
            if not docs and not vids:
                st.info("Nothing added to this subject yet.")
            if docs:
                st.markdown("#### PDFs, handwritten pages and YouTube")
                for d in docs:
                    c1, c2 = st.columns([8, 1])
                    c1.write(f"📄 **{d['doc']}**  ·  type: {d['kind']}  ·  {d['n']} chunks")
                    with c2.popover("🗑"):
                        st.write("This file and its data will be deleted.")
                        if st.button("Yes, delete", key=f"deldoc_{active}_{d['doc']}"):
                            remove_doc(active, d["doc"], d["kind"])
                            st.rerun()
            if vids:
                st.markdown("#### Videos")
                hl_store = st.session_state.setdefault("video_highlights", {})
                notes_store = st.session_state.setdefault("video_notes", {})
                for v in vids:
                    name, analysis = v["name"], v["analysis"]
                    key = f"{active}::{name}"
                    with st.expander(f"🎬 {name}"):
                        with st.popover("🗑 Remove this video"):
                            st.write("This video and its analysis will be deleted.")
                            if st.button("Yes, delete", key=f"delvid_{key}"):
                                remove_video(active, name)
                                st.rerun()
                        if hl_store.get(key):
                            st.markdown("**Highlights**")
                            st.markdown(hl_store[key])
                        elif st.button("Make highlights", key=f"hl_{key}"):
                            try:
                                with st.spinner("Finding the highlights..."):
                                    hl_store[key] = make_highlights(analysis)
                                db.save_generated(active, f"Highlights — {name}", "notes",
                                                  hl_store[key], source=name)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not write highlights: {e}")

                        with st.expander("Full timeline (everything the model saw and heard)"):
                            st.markdown(analysis)

                        st.divider()
                        style = st.selectbox("Notes format", list(FORMATS), key=f"style_{key}")
                        lang = st.selectbox("Language", ["Same as the video", "English", "Hindi"],
                                            key=f"lang_{key}")
                        if st.button("Make notes", key=f"notes_{key}"):
                            try:
                                with st.spinner("Writing notes..."):
                                    notes_store[key] = make_notes(analysis, style, lang)
                                db.save_generated(active, f"{style} — {name}", "notes",
                                                  notes_store[key], source=name)
                            except Exception as e:
                                st.error(f"Could not write notes: {e}")
                        if notes_store.get(key):
                            st.markdown(notes_store[key])
                            st.download_button("Download notes (.md)", notes_store[key],
                                               file_name=f"{name}_notes.md", key=f"dl_{key}")

        with tab_gen:
            if not gens:
                st.info("Nothing generated yet. Notes, mind maps, flowcharts and highlights "
                        "you create in Chat are saved here automatically.")
            for g in gens:
                icon = "🧠" if g["kind"] == "diagram" else "📝"
                with st.expander(f"{icon} {g['title']}  ·  {g['created']}"):
                    if g.get("png"):
                        st.image(g["png"])
                        st.download_button("⬇ Download image", g["png"],
                                           file_name=f"{g['title']}.png", mime="image/png",
                                           key=f"gdlp_{g['id']}")
                    else:
                        st.markdown(g["content"])
                        st.download_button("⬇ Download (.md)", g["content"],
                                           file_name=f"{g['title']}.md", key=f"gdlm_{g['id']}")
                    with st.popover("🗑 Delete"):
                        st.write("This item will be deleted.")
                        if st.button("Yes, delete", key=f"delgen_{g['id']}"):
                            db.delete_generated(g["id"])
                            st.rerun()
