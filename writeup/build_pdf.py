"""Build writeup/report.pdf from writeup/report.md. Run from the repo root.

Needs: markdown-it-py, playwright, Google Chrome, and internet access (MathJax).
"""
import html
import re
from pathlib import Path

from markdown_it import MarkdownIt
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent

source = (HERE / "report.md").read_text()

# Pull math out first so Markdown does not treat _ and \ inside it as formatting.
# Code blocks and code spans are matched too, and left untouched.
token = re.compile(r"(```.*?```|`[^`\n]*`)|\$\$(.+?)\$\$|\$([^$\n]+?)\$", re.S)
math = []


def stash(m):
    if m.group(1):
        return m.group(1)
    display = m.group(2) is not None
    body = html.escape(m.group(2) if display else m.group(3))
    math.append(f"\\[{body}\\]" if display else f"\\({body}\\)")
    return f"@@MATH{len(math) - 1}@@"


body = MarkdownIt("commonmark").enable("table").render(token.sub(stash, source))
body = re.sub(r"@@MATH(\d+)@@", lambda m: math[int(m.group(1))], body)

CSS = """
body { font: 10.5pt/1.4 Georgia, 'Times New Roman', serif; color: #111; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 2pt; } h2 { font-size: 13.5pt; }
h2:first-of-type { margin-top: 2pt; }
h3 { font-size: 11.5pt; } h1, h2, h3 { break-after: avoid; line-height: 1.25; }
code { font: 9pt Menlo, monospace; } pre { background: #f4f4f4; padding: 6pt 8pt; break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9.5pt; break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 6pt; text-align: center; } th { background: #eee; }
img { display: block; margin: 6pt auto; max-width: 85%; max-height: 3.4in; break-inside: avoid; }
td img { max-width: 2.5in; }
p:has(+ p > img) { break-after: avoid; }
p:has(> img) { break-inside: avoid; }
hr { border: 0; border-top: 1px solid #ccc; }
"""
page_html = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script></head>
<body>{body}</body></html>"""

tmp = HERE / "_report_build.html"
tmp.write_text(page_html)
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.goto(tmp.as_uri(), wait_until="networkidle")
        page.evaluate("() => MathJax.startup.promise")
        errors = page.eval_on_selector_all("mjx-merror, [data-mjx-error]", "e => e.length")
        broken = page.eval_on_selector_all("img", "e => e.filter(i => !i.naturalWidth).length")
        page.pdf(
            path=str(HERE / "report.pdf"), format="Letter", print_background=True,
            margin={"top": "0.8in", "bottom": "0.8in", "left": "0.85in", "right": "0.85in"},
            display_header_footer=True, header_template="<span></span>",
            footer_template='<div style="font-size:8px;width:100%;text-align:center">'
                            '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
        )
        browser.close()
finally:
    tmp.unlink()
print(f"math expressions: {len(math)}, math errors: {errors}, broken images: {broken}")
