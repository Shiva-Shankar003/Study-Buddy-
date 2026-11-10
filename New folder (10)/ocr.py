import os
import cv2
import numpy as np
import config

_reader = None

def deskew(bw, max_angle=10.0, step=0.5):
    ink = cv2.resize(255 - bw, None, fx=0.5, fy=0.5)
    h, w = ink.shape
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        m = cv2.getRotationMatrix2D((w/2, h/2), float(angle), 1.0)
        rotated = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    wo, ho = bw.shape[1], bw.shape[0]
    m = cv2.getRotationMatrix2D((wo/2, ho/2), best_angle, 1.0)
    return cv2.warpAffine(bw, m, (wo, ho), flags=cv2.INTER_CUBIC, borderValue=255)

def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return deskew(bw)

def ocr_easyocr(path):
    global _reader
    import easyocr
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    lines = _reader.readtext(preprocess(path), detail=0, paragraph=True)
    return "\n".join(lines)

def ocr_tesseract(path):
    import pytesseract
    return pytesseract.image_to_string(preprocess(path))

def ocr_gemini(path):
    from google import genai
    from PIL import Image
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    prompt = "Transcribe all handwritten text in this image exactly as written. Return only the text."
    resp = client.models.generate_content(model=config.GEMINI_MODEL, contents=[Image.open(path), prompt])
    return resp.text or ""

ENGINES = {"easyocr": ocr_easyocr, "tesseract": ocr_tesseract, "gemini": ocr_gemini}

def load_handwritten(path, engine=None, page=1):
    engine_func = ENGINES.get(engine or config.OCR_ENGINE, ocr_easyocr)
    text = engine_func(path).strip()
    if len(text) < 10:
        return []
    name = os.path.basename(path)
    return [{"loc": page, "text": text, "ref": f"{name} (handwritten, p.{page})"}]