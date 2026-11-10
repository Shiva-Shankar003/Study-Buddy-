"""Analyze a whole video, then make highlights and notes from the analysis.

 analyze_with_gemini(path): Gemini watches the video (pictures + audio). Needs GEMINI_API_KEY.
 analyze_locally(path):     free, local: speech (Whisper) + text shown on screen (OCR).
 make_highlights(analysis): overview + the most important moments.
 make_notes(analysis, style, language): notes in the format the student picks.
"""
import time

from llm import ask

TIMELINE_PROMPT = (
    "Watch this whole video carefully, looking at everything shown and listening to everything said. "
    "Write a detailed timeline. For every part of roughly 30 to 60 seconds, start the line with the "
    "time like [mm:ss], then describe what is shown on screen (slides, board, diagrams, demos, "
    "text, people, actions) and what is being explained. Be factual and do not invent anything. "
    "Write in the language of the speaker. Finish with one line starting with TOPIC: that names "
    "the overall topic."
)

FORMATS = {
    "Short summary (5-8 lines)": "a short summary of 5 to 8 lines",
    "Detailed notes (headings + points)": "detailed study notes with headings and bullet points, "
                                          "keeping every important definition, formula, step and example",
    "Bullet points only": "only crisp bullet points, grouped under short headings",
    "Exam revision sheet": "an exam revision sheet: key points, formulas or definitions to memorise, "
                           "and 5 likely exam questions with short answers",
    "Step-by-step explanation": "a step-by-step explanation in the order things happen in the video",
}


def _state_name(f):
    return str(getattr(getattr(f, "state", None), "name", getattr(f, "state", "")))


def analyze_with_gemini(path):
    import config
    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    f = client.files.upload(file=path)

    waited = 0
    while _state_name(f) == "PROCESSING":
        time.sleep(5)
        waited += 5
        if waited > 900:
            raise RuntimeError("Gemini took too long to process the video")
        f = client.files.get(name=f.name)
    if _state_name(f) == "FAILED":
        raise RuntimeError("Gemini could not process this video")

    model = getattr(config, "GEMINI_MODEL", None) or "gemini-flash-latest"
    resp = client.models.generate_content(model=model, contents=[f, TIMELINE_PROMPT])
    try:
        client.files.delete(name=f.name)
    except Exception:
        pass

    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty analysis")
    return text


def analyze_locally(path, read_screen=True):
    import video_loader

    units = video_loader.load_video(path, window=45, read_screen=read_screen)
    if not units:
        raise RuntimeError("No speech or readable text was found in this video")
    return "\n".join(f"[{u['ref'].rsplit(', ', 1)[-1]}] {u['text']}" for u in units)


def make_highlights(analysis):
    prompt = (
        "Below is a timestamped analysis of a video.\n"
        "Write:\n"
        "OVERVIEW: 2 to 3 simple sentences on what the video is about.\n"
        "HIGHLIGHTS: the 6 to 10 most important moments, one per line, as "
        "'[mm:ss] short title - why it matters'.\n"
        "Use only the analysis. Use the same language as the analysis.\n\n"
        "Analysis:\n" + analysis[:60000]
    )
    return ask(prompt, system="You summarise videos for students accurately.").strip()


def make_notes(analysis, style, language="Same as the video"):
    wanted = FORMATS.get(style, "clear study notes")
    lang = ("Write in the same language as the analysis." if language.startswith("Same")
            else f"Write in {language}.")
    prompt = (
        f"Make {wanted} from this timestamped video analysis. Keep timestamps like [mm:ss] "
        f"next to important points when useful. Use only what is in the analysis. {lang}\n\n"
        "Analysis:\n" + analysis[:60000]
    )
    return ask(prompt, system="You write accurate, easy-to-study notes for students.").strip()
