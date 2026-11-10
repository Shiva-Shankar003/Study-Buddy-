import os
import pymupdf

def load_pdf(path):
    name = os.path.basename(path)
    units = []
    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if len(text) < 30:
                continue
            units.append({"loc": number, "text": text, "ref": f"{name}, p.{number}"})
    return units