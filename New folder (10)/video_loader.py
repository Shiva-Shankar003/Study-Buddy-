"""Turn an uploaded video file into text units (same format as yt_loader.load_youtube).

 - Speech is transcribed locally with faster-whisper (no API key, no YouTube).
 - Optionally, text shown on screen (slides, board) is read with OCR every few seconds.
Each unit: {"loc": seconds, "text": "...", "ref": "<file name>, mm:ss"}
"""
import os

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        size = os.getenv("SB_WHISPER_MODEL", "base")
        _model = WhisperModel(size, device="cpu", compute_type="int8")
    return _model


def _stamp(seconds):
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def transcribe(path, language=None):
    """Return [(start_seconds, text), ...] for the speech in the video."""
    segments, _info = _get_model().transcribe(path, language=language, vad_filter=True)
    return [(float(s.start), s.text.strip()) for s in segments if s.text.strip()]


def read_screen_text(path, every=30, max_frames=60):
    """Return [(seconds, '[On screen] text'), ...] using OCR on sampled frames."""
    import cv2
    import tempfile
    from ocr import ENGINES

    engine = ENGINES.get("easyocr") or next(iter(ENGINES.values()))
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(fps * every))
    items, last_words = [], set()
    for k, frame_no in enumerate(range(0, total, step)):
        if k >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            break
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        cv2.imwrite(tmp.name, frame)
        try:
            text = (engine(tmp.name) or "").strip()
        except Exception:
            text = ""
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass
        words = set(text.lower().split())
        if len(words) < 3:
            continue
        overlap = len(words & last_words) / max(1, len(words | last_words))
        if overlap > 0.8:  # same slide as before
            continue
        last_words = words
        items.append((frame_no / fps, "[On screen] " + text))
    cap.release()
    return items


def _group(pairs, name, window):
    units, buf, start = [], [], 0.0

    def flush():
        if buf:
            units.append({"loc": int(start), "text": " ".join(buf),
                          "ref": f"{name}, {_stamp(start)}"})

    for s, text in pairs:
        if not buf:
            start = s
        buf.append(text.replace("\n", " "))
        if s - start >= window:
            flush()
            buf = []
    flush()
    return units


def load_video(path, window=60, read_screen=False, language=None):
    name = os.path.basename(path)
    pairs = transcribe(path, language)
    if read_screen:
        pairs += read_screen_text(path)
    pairs.sort(key=lambda p: p[0])
    return _group(pairs, name, window)
