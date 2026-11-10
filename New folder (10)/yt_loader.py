import re
from youtube_transcript_api import YouTubeTranscriptApi


def video_id(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", url)
    if not m:
        raise ValueError(f"Could not find a YouTube video id in: {url}")
    return m.group(1)


def _group(pairs, vid, window):
    units, buf, start = [], [], 0.0

    def flush():
        if buf:
            t = int(start)
            units.append({"loc": t, "text": " ".join(buf),
                          "ref": f"https://youtu.be/{vid}?t={t}"})

    for s, text in pairs:
        if not buf:
            start = s
        buf.append(text.replace("\n", ""))
        if s - start >= window:
            flush()
            buf = []
    flush()
    return units


def load_youtube(url, window=60):
    vid = video_id(url)
    tlist = YouTubeTranscriptApi().list(vid)
    try:
        transcript = tlist.find_transcript(["en", "hi"])
    except Exception:
        transcript = next(iter(tlist))  # कोई भी उपलब्ध भाषा ले लो
    snippets = transcript.fetch()
    return _group([(s.start, s.text) for s in snippets], vid, window)