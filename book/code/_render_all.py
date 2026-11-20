"""
Internal helper (not a chapter script): runs every chapter script headlessly
and saves each matplotlib figure it would have shown to book/images/ instead.
Not part of the course content -- just the tool used to generate the book's images.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import runpy
import os
import sys
import traceback

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.dirname(CODE_DIR)
IMAGES_DIR = os.path.join(BOOK_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

state = {"chapter": None, "count": 0}

def fake_show(*args, **kwargs):
    chapter = state["chapter"]
    for num in plt.get_fignums():
        fig = plt.figure(num)
        state["count"] += 1
        path = os.path.join(IMAGES_DIR, f"{chapter}_fig{state['count']}.png")
        fig.savefig(path, dpi=115, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close("all")

plt.show = fake_show

scripts = sorted(
    f for f in os.listdir(CODE_DIR)
    if f.endswith(".py") and not f.startswith("_")
)

for script in scripts:
    chapter_name = script[:-3]
    state["chapter"] = chapter_name
    state["count"] = 0
    print(f"Running {script} ...")
    try:
        runpy.run_path(os.path.join(CODE_DIR, script), run_name="__main__")
    except Exception:
        print(f"  !! FAILED: {script}")
        traceback.print_exc()
    plt.close("all")

print("Done.")
