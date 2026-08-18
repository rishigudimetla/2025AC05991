"""
Build the single submission PDF required by ML Assignment 2, in the mandated order:

    1. GitHub repository link
    2. Live Streamlit app link
    3. Screenshot of the assignment executing on BITS Virtual Lab
    4. The full README.md content

Toolchain: pandoc (Markdown -> HTML) + headless Chrome (HTML -> PDF). Both are already
present on macOS installs of Chrome + Homebrew pandoc; no LaTeX needed.

Usage:
    python tools/make_submission_pdf.py \
        --repo https://github.com/<user>/<repo> \
        --app  https://<app>.streamlit.app \
        --screenshot ~/Desktop/bits_lab.png \
        --student "Rishi Gudimetla" --id 20XXXXXXXX
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_HTML = ROOT / "build" / "submission.html"
OUT_PDF = ROOT / "build" / "ML_Assignment_2_Submission.pdf"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Times New Roman", Georgia, serif;
  font-size: 10.5pt; line-height: 1.45; color: #000; margin: 0;
}
h1 { font-size: 17pt; margin: 0 0 .4em; font-weight: bold; }
h2 { font-size: 13pt; margin: 1.4em 0 .5em; font-weight: bold;
     border-bottom: 1px solid #000; padding-bottom: .15em; }
h3 { font-size: 11.5pt; margin: 1.1em 0 .35em; font-weight: bold; }
h4 { font-size: 10.5pt; margin: .9em 0 .3em; font-weight: bold; }
a { color: #0645ad; text-decoration: underline; word-break: break-all; }
code { font-family: "Courier New", monospace; font-size: 9pt; }
pre { border: 1px solid #999; padding: .5em .7em; overflow-wrap: break-word;
      white-space: pre-wrap; background: #fff; }
pre code { font-size: 8.4pt; }
table { border-collapse: collapse; width: 100%; margin: .6em 0; font-size: 8.8pt; }
th, td { border: 1px solid #000; padding: 3px 5px; text-align: left; vertical-align: top; }
th { font-weight: bold; }
blockquote { margin: .7em 0 .7em 1em; padding-left: .8em; border-left: 2px solid #000; }
hr { border: 0; border-top: 1px solid #000; margin: 1.2em 0; }
.cover { page-break-after: always; }
.cover h1 { text-align: center; font-size: 18pt; margin-bottom: .15em; }
.cover .sub { text-align: center; font-size: 11pt; margin: 0 0 .2em; }
.cover .dept { text-align: center; font-size: 10pt; margin: 0 0 1.6em; }
.meta table td:first-child { width: 32%; font-weight: bold; }
.linkbox { margin: 10px 0 14px; }
.linkbox .label { font-weight: bold; }
.linkbox .url { word-break: break-all; }
.linkbox .url a { color: #0645ad; text-decoration: underline; }
.idhdr { font-weight: bold; font-size: 12pt; margin: 0 0 14px; }
.linkbox .hint { font-size: 9pt; margin-top: 2px; }
.shot { page-break-inside: avoid; margin-top: .5em; }
.shot img { width: 100%; border: 1px solid #000; }
.todo { border: 1px solid #000; padding: 22px 16px; text-align: center; font-weight: bold; }
.caption { font-size: 9pt; margin-top: 4px; font-style: italic; }
.section-break { page-break-before: always; }
.idrow { width: 100%; border: 0; margin: 0 0 6mm; }
.idrow td { border: 0; padding: 0; font-weight: bold; font-size: 13pt; }
.idrow td.r { text-align: right; }
h1.ctitle { text-align: center; font-size: 19pt; margin: 0 0 9mm; font-weight: bold; }
.item { margin: 0; }
.num { display: inline-block; width: 8mm; font-weight: bold; }
.lbl { font-weight: bold; }
.urlline { margin: 0 0 6mm 8mm; }
.urlline a { color: #0645ad; text-decoration: underline; }
.headrule { border-bottom: 1px solid #000; padding-bottom: 1.2mm; margin: 0 0 3mm; }
.subhead { font-weight: bold; border-bottom: 1px solid #000;
           padding-bottom: 1.2mm; margin: 5mm 0 3mm; }
.coverimg { width: 148mm; display: block; margin: 0 auto; border: 1px solid #555; }
"""


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if candidate and Path(candidate).exists():
            return candidate
    sys.exit("No Chrome/Chromium/Edge found - cannot render the PDF.")


def embed_image(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def readme_html() -> str:
    readme = ROOT / "README.md"
    if not readme.exists():
        sys.exit("README.md not found - run tools/make_readme.py first.")
    result = subprocess.run(
        ["pandoc", "--from=gfm", "--to=html5", str(readme)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def build_html(repo: str, app: str, shot: Path | None, student: str, sid: str,
               app_shot: Path | None = None) -> str:
    today = date.today().strftime("%d %B %Y")

    if shot and shot.exists():
        shot_block = f'<img class="coverimg" src="{embed_image(shot)}" alt="BITS Virtual Lab execution"/>' 
    else:
        shot_block = (
            '<div class="todo">SCREENSHOT PLACEHOLDER<br/>'
            "Re-run this generator with <code>--screenshot &lt;path-to-png&gt;</code> "
            "to embed the BITS Virtual Lab screenshot.</div>"
        )

    if app_shot and app_shot.exists():
        app_shot_block = f'<img class="coverimg" src="{embed_image(app_shot)}" alt="Deployed app"/>' 
    else:
        app_shot_block = ""

    def item(n: int, label: str, url: str) -> str:
        shown = url if "REPLACE-ME" not in url else f"{url} (replace before submitting)"
        return (
            f'<div class="item"><span class="num">{n}.</span>'
            f'<span class="lbl">{label}</span></div>'
            f'<div class="urlline"><a href="{url}">{shown}</a></div>'
        )

    app_shot_section = ("" if not app_shot_block else
        '<div class="subhead">Deployed Streamlit application</div>' + app_shot_block)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ML Assignment 2 - Submission</title>
<style>{CSS}</style></head><body>

<div class="cover">
  <table class="idrow"><tr>
    <td class="l">RishiGudimetla</td>
    <td class="r">{sid}</td>
  </tr></table>

  <h1 class="ctitle">Machine Learning - Assignment 2</h1>

  {item(1, "GitHub Repository Link:", repo)}
  {item(2, "Live Streamlit App Link:", app)}

  <div class="headrule"><span class="num">3.</span><span class="lbl">Screenshot</span>: execution on BITS Virtual Lab</div>
  {shot_block}

  {app_shot_section}
</div>

<div class="section-break">
  <h1>4. README.md content</h1>
  <p class="caption">Reproduced in full from the repository, as required by Section 2, item 4.</p>
  <hr/>
  {readme_html()}
</div>

</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="https://github.com/REPLACE-ME/ml-assignment2-student-outcome-radar")
    ap.add_argument("--app", default="https://REPLACE-ME.streamlit.app")
    ap.add_argument("--screenshot", default=None, help="path to the BITS Virtual Lab screenshot")
    ap.add_argument("--app-screenshot", dest="app_screenshot", default=None,
                    help="optional screenshot of the deployed app (appendix)")
    ap.add_argument("--student", default="Rishi Gudimetla")
    ap.add_argument("--id", dest="sid", default="<student ID>")
    args = ap.parse_args()

    shot = Path(args.screenshot).expanduser() if args.screenshot else None
    if args.screenshot and not shot.exists():
        sys.exit(f"Screenshot not found: {shot}")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    app_shot = Path(args.app_screenshot).expanduser() if args.app_screenshot else None
    if args.app_screenshot and not app_shot.exists():
        sys.exit(f"App screenshot not found: {app_shot}")
    OUT_HTML.write_text(build_html(args.repo, args.app, shot, args.student, args.sid, app_shot))
    print(f"Wrote {OUT_HTML.relative_to(ROOT)}")

    chrome = find_chrome()
    subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()],
        check=True, capture_output=True,
    )
    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"Wrote {OUT_PDF.relative_to(ROOT)} ({size_kb:.0f} KB)")

    if "REPLACE-ME" in args.repo or "REPLACE-ME" in args.app or shot is None:
        print("\nStill to do before submitting:")
        if "REPLACE-ME" in args.repo:
            print("  - pass the real --repo URL")
        if "REPLACE-ME" in args.app:
            print("  - pass the real --app URL")
        if shot is None:
            print("  - pass --screenshot <BITS Virtual Lab screenshot>")


if __name__ == "__main__":
    main()
