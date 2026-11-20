"""
Internal helper (not a chapter script, one-time fix): the embedded lab
modules inside chapter Markdown run as soon as the browser parses them
(they're inline <script> tags in the middle of the article), but app.js
(which defines window.StatsLab) is only loaded near the bottom of the page.
So StatsLab was undefined when each module's script ran, it threw silently,
and nothing got wired up. This wraps every such module's logic in a
DOMContentLoaded listener so it runs only after app.js has finished loading.
"""
import os
import re

CHAPTERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chapters")

OPEN_OLD = "(function(){\n  var SL = window.StatsLab;"
OPEN_NEW = "document.addEventListener('DOMContentLoaded', function(){\n  var SL = window.StatsLab;"

for fname in sorted(os.listdir(CHAPTERS_DIR)):
    if not fname.endswith(".md"):
        continue
    path = os.path.join(CHAPTERS_DIR, fname)
    with open(path, encoding="utf-8") as f:
        text = f.read()

    if OPEN_OLD not in text:
        continue

    n_open = text.count(OPEN_OLD)
    new_text = text.replace(OPEN_OLD, OPEN_NEW)

    # The matching closer for our modules is always "})();\n</script>" --
    # verify the count matches before touching anything, to avoid mangling
    # unrelated script blocks.
    n_close = new_text.count("})();\n</script>")
    if n_close < n_open:
        print(f"WARNING: {fname} has {n_open} module(s) but only {n_close} matching closers -- skipped, check manually")
        continue

    new_text = new_text.replace("})();\n</script>", "});\n</script>")

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"fixed {fname}: {n_open} module(s)")

print("\nDone.")
