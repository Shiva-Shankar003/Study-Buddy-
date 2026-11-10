import glob
import os
import sys

import jiwer

from ocr import ENGINES


def norm(text):
    return " ".join(text.lower().split())


def main(folder):
    """folder holds pairs like page1.jpg + page1.txt (the txt is your hand-typed ground truth)."""
    scores = {name: {"cer": [], "wer": []} for name in ENGINES}
    images = sorted(glob.glob(os.path.join(folder, "*.jpg"))
                    + glob.glob(os.path.join(folder, "*.png")))
    for img in images:
        truth = norm(open(os.path.splitext(img)[0] + ".txt", encoding="utf-8").read())
        for name, engine in ENGINES.items():
            try:
                guess = norm(engine(img))
            except Exception as e:
                print(f"{name} failed on {os.path.basename(img)}: {e}")
                continue
            scores[name]["cer"].append(jiwer.cer(truth, guess))
            scores[name]["wer"].append(jiwer.wer(truth, guess))
    for name, s in scores.items():
        if s["cer"]:
            print(f"{name:10s} CER={sum(s['cer']) / len(s['cer']):.3f} "
                  f"WER={sum(s['wer']) / len(s['wer']):.3f} pages={len(s['cer'])}")


if __name__ == "__main__":
    main(sys.argv[1])
