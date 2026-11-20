"""
Internal helper (not a chapter script): builds a real static HTML site from
the book's Markdown chapters -- the "Jekyll-style" pipeline. Produces a
reading app with:
  - a persistent left sidebar with the FULL table of contents (grouped by
    part, current chapter highlighted, with a quick filter box) on every page
  - the chapter text in the middle
  - a sticky image rail on the right, synced to scroll position
  - text-highlighting with colored popup notes, persisted in localStorage
  - MathJax for formulas
On narrow/mobile screens the sidebar becomes a slide-in drawer (hamburger
button) and the image rail collapses, with images shown inline instead.

Run with:  python3 book/code/_build_site.py
Then serve with:  python3 -m http.server 8000 --directory book/site
and open:  http://localhost:8000/
"""
import os
import re
import shutil
import markdown

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # book/
CHAPTERS_DIR = os.path.join(BASE, "chapters")
IMAGES_DIR = os.path.join(BASE, "images")
ASSETS_DIR = os.path.join(BASE, "site_assets")
SITE_DIR = os.path.join(BASE, "site")

BLOCK_TAGS = ("p", "li", "blockquote", "h2", "h3")

_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)")


def protect_math(text):
    """Pull every $$...$$ and $...$ span out of the raw Markdown BEFORE
    conversion, replacing each with an inert alphanumeric placeholder, and
    return (protected_text, {placeholder: original_span}).

    Why this is necessary: Markdown's emphasis parser treats underscores and
    asterisks as formatting markers. A formula like $$\\underbrace{P(H\\mid
    E)}_{\\text{posterior}} = ... \\underbrace{P(E)}_{\\text{evidence}}$$ has
    TWO underscores that aren't surrounded by word characters, so Markdown
    pairs them up as <em>...</em> and silently deletes the underscores --
    corrupting the LaTeX before MathJax ever sees it. Extracting math first
    and restoring it verbatim after conversion sidesteps this entirely.
    """
    placeholders = {}
    counter = {"n": 0}

    def stash(match):
        counter["n"] += 1
        key = "XMATHPLACEHOLDER{0}X".format(counter["n"])
        placeholders[key] = match.group(0)
        return key

    text = _DISPLAY_MATH_RE.sub(stash, text)
    text = _INLINE_MATH_RE.sub(stash, text)
    return text, placeholders


def restore_math(html_body, placeholders):
    for key, original in placeholders.items():
        html_body = html_body.replace(key, original)
    return html_body

# (part title, inclusive range of chapter-number prefixes)
PARTS = [
    ("Orientation", range(0, 1)),
    ("Part I — Describing the World", range(1, 9)),
    ("Part II — Inferring From Data", range(9, 16)),
    ("Part III — The Bridge Into Machine Learning", range(16, 21)),
    ("Part IV — The Statistics Inside Modern AI", range(21, 27)),
    ("Part V — The Statistical Frontier of Modern AI", range(27, 28)),
]

PAGE_SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{root}assets/app.css">
<script>
window.MathJax = {{ tex: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']], processEscapes: true }} }};
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"></script>
</head>
<body data-chapter="{chapter_id}">
<div class="topbar">
  <div class="left">
    <button class="iconbtn hamburger" id="btnSidebarToggle">☰</button>
    <a href="{root}index.html">Statistics for AI</a>
  </div>
  <div class="right">
    <button class="iconbtn" id="btnNotesPanel">📒 My Notes</button>
  </div>
</div>
<div id="sidebarOverlay"></div>
<div class="layout">
  <nav class="sidebar" id="sidebar">
    <input type="text" id="sidebarFilter" placeholder="Filter chapters...">
{sidebar_parts}
  </nav>
  <div class="pane-left">
    <div class="article" id="article">
{body}
    </div>
    <div class="navfooter">
      <span>{prev_link}</span>
      <span>{next_link}</span>
    </div>
  </div>
  <div class="pane-right">
    <h4>Figures in this chapter</h4>
    <div id="railContent"></div>
  </div>
</div>

<!-- floating selection toolbar -->
<div id="selToolbar">
  <button data-color="yellow" title="Mark interesting">🟡</button>
  <button data-color="green" title="Mark understood">🟢</button>
  <button data-color="red" title="Mark confusing">🔴</button>
  <button data-color="blue" title="Mark to review later">🔵</button>
  <button id="btnAddNote" title="Highlight + add a note">📝</button>
</div>

<!-- note edit popup -->
<div id="notePopup">
  <textarea placeholder="Write a note to yourself..."></textarea>
  <div class="row">
    <button class="danger" id="btnDeleteHl">Remove highlight</button>
    <div>
      <button id="btnCloseNote">Close</button>
      <button class="primary" id="btnSaveNote">Save</button>
    </div>
  </div>
</div>

<!-- notes side panel -->
<div id="notesOverlay"></div>
<div id="notesPanel">
  <div class="header">
    <b>My Notes — this page</b>
    <button class="iconbtn" id="btnCloseNotesPanel">✕</button>
  </div>
  <div class="list" id="notesList"></div>
  <div class="footer">
    <button id="btnExportNotes">⬇ Export all notes</button>
    <button id="btnImportNotes">⬆ Import notes</button>
    <input type="file" id="importFile" accept="application/json" style="display:none;">
  </div>
</div>

<script src="{root}assets/app.js"></script>
</body>
</html>
"""


def inject_block_ids(html_body):
    """Add a stable data-bid="b<N>" to every prose block element, so
    highlights/notes can be persisted and re-attached across reloads."""
    counter = {"n": 0}

    def repl(match):
        tag = match.group(1)
        rest = match.group(2)
        counter["n"] += 1
        return '<{0} data-bid="b{1}"{2}>'.format(tag, counter["n"], rest)

    pattern = re.compile(r"<(" + "|".join(BLOCK_TAGS) + r")([^>]*)>")
    return pattern.sub(repl, html_body)


def mark_figures_for_rail(html_body):
    """Tag each image (and its following *caption* paragraph, if any) so the
    client-side script can ALSO clone it into the right-hand rail as a
    scroll-synced quick reference. The image itself always stays inline in
    the article text -- this never hides or removes it, only adds a copy."""
    fig_counter = {"n": 0}

    def repl(match):
        img_tag = match.group(1)
        caption_block = match.group(2) or ""
        fig_counter["n"] += 1
        fig_id = "f{0}".format(fig_counter["n"])

        img_tag2 = re.sub(r"^<img ", '<img class="has-rail" data-figid="{0}" '.format(fig_id), img_tag)

        if caption_block:
            caption_block2 = re.sub(
                r"^<p([^>]*)>", r'<p\1 class="img-caption has-rail" data-figid="{0}">'.format(fig_id),
                caption_block
            )
        else:
            caption_block2 = ""

        return img_tag2 + caption_block2

    pattern = re.compile(r"<p>\s*(<img [^>]+/?>)\s*</p>\s*(<p[^>]*><em>.*?</em></p>)?", re.DOTALL)
    return pattern.sub(repl, html_body)


def build_sidebar(titles, md_files, current_fname, chapter_href_prefix):
    """titles: dict fname -> title. chapter_href_prefix: '' when rendering
    chapter pages (siblings in the same folder), 'chapters/' for index.html."""
    out = []
    for part_title, num_range in PARTS:
        items = [f for f in md_files if int(f[:2]) in num_range]
        if not items:
            continue
        out.append('<details open><summary>{0}</summary><ul>'.format(part_title))
        for fname in items:
            is_current = "current" if fname == current_fname else ""
            href = chapter_href_prefix + fname.replace(".md", ".html")
            title = titles[fname]
            data_title = title.lower().replace('"', "")
            out.append(
                '<li class="{0}" data-title="{1}"><a href="{2}">{3}</a></li>'.format(
                    is_current, data_title, href, title
                )
            )
        out.append("</ul></details>")
    return "\n".join(out)


def build():
    if os.path.exists(SITE_DIR):
        shutil.rmtree(SITE_DIR)
    os.makedirs(os.path.join(SITE_DIR, "chapters"))
    shutil.copytree(IMAGES_DIR, os.path.join(SITE_DIR, "images"))
    shutil.copytree(ASSETS_DIR, os.path.join(SITE_DIR, "assets"))

    md_files = sorted(f for f in os.listdir(CHAPTERS_DIR) if f.endswith(".md"))

    # First pass: cheaply pull each chapter's title straight from its first
    # markdown heading, so the full sidebar can be built before rendering.
    titles = {}
    for fname in md_files:
        with open(os.path.join(CHAPTERS_DIR, fname), encoding="utf-8") as f:
            first_text = f.read()
        m = re.search(r"^# (.+)$", first_text, re.MULTILINE)
        titles[fname] = m.group(1).strip() if m else fname

    md = markdown.Markdown(extensions=["extra", "fenced_code", "sane_lists", "toc"])

    for i, fname in enumerate(md_files):
        md.reset()
        with open(os.path.join(CHAPTERS_DIR, fname), encoding="utf-8") as f:
            text = f.read()

        protected_text, placeholders = protect_math(text)
        html_body = md.convert(protected_text)
        html_body = restore_math(html_body, placeholders)
        html_body = mark_figures_for_rail(html_body)
        html_body = inject_block_ids(html_body)

        title = titles[fname]
        chapter_id = fname[:-3]

        out_name = fname.replace(".md", ".html")
        prev_link = f'<a href="{md_files[i-1].replace(".md",".html")}">&larr; Previous chapter</a>' if i > 0 else ""
        next_link = f'<a href="{md_files[i+1].replace(".md",".html")}">Next chapter &rarr;</a>' if i < len(md_files)-1 else ""

        sidebar_parts = build_sidebar(titles, md_files, fname, chapter_href_prefix="")

        page = PAGE_SHELL.format(title=title, body=html_body, root="../", chapter_id=chapter_id,
                                  sidebar_parts=sidebar_parts,
                                  prev_link=prev_link, next_link=next_link)
        out_path = os.path.join(SITE_DIR, "chapters", out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(page)

        print("built", out_name)

    # Landing page uses the exact same shell (sidebar, rail, notes panel),
    # just with a welcome blurb as the article body and no figures.
    welcome_body = """
<h1>Statistics for AI — A Book From Zero to Deep Learning Foundations</h1>
<p data-bid="idx1">Pick any chapter from the sidebar on the left to start reading. Many chapters have a
live, interactive module built right into the text wherever a concept benefits from it — drag a slider and
watch the math recompute. Select any text to highlight it or leave yourself a note (saved locally in this
browser) — click the 📒 My Notes button any time to review everything you've marked.</p>
<p data-bid="idx2">New here? Start with <b>Chapter 0</b> at the top of the sidebar.</p>
"""
    sidebar_parts = build_sidebar(titles, md_files, current_fname=None, chapter_href_prefix="chapters/")
    index_page = PAGE_SHELL.format(title="Statistics for AI", body=welcome_body, root="", chapter_id="__index__",
                                    sidebar_parts=sidebar_parts, prev_link="", next_link="")
    with open(os.path.join(SITE_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_page)
    print("built index.html")

    print(f"\nSite built at {SITE_DIR}")
    print("Serve it with:  python3 -m http.server 8000 --directory \"%s\"" % SITE_DIR)
    print("Then open:      http://localhost:8000/")


if __name__ == "__main__":
    build()
