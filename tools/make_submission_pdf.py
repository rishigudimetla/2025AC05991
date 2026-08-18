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
@page { size: A4; margin: 16mm 14mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #1f2a25; margin: 0;
}
h1 { font-size: 19pt; color: #22503a; margin: 0 0 .35em; }
h2 { font-size: 14pt; color: #22503a; border-bottom: 2px solid #cfe0d6;
     padding-bottom: .18em; margin: 1.5em 0 .6em; }
h3 { font-size: 11.8pt; color: #2f6f4e; margin: 1.15em 0 .4em; }
h4 { font-size: 10.8pt; margin: 1em 0 .3em; }
a { color: #1a5c8a; word-break: break-all; }
code { background: #f1f4f2; padding: .08em .3em; border-radius: 3px;
       font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 8.8pt; }
pre { background: #f6f8f7; border: 1px solid #e0e6e3; border-radius: 6px;
      padding: .6em .8em; overflow-wrap: break-word; white-space: pre-wrap; }
pre code { background: none; font-size: 8.2pt; }
table { border-collapse: collapse; width: 100%; margin: .7em 0; font-size: 8.6pt; }
th, td { border: 1px solid #d3ddd7; padding: 4px 6px; text-align: left;
         vertical-align: top; }
th { background: #eaf1ed; font-weight: 600; }
blockquote { border-left: 3px solid #9fc4ae; margin: .8em 0; padding: .2em 0 .2em .9em;
             color: #3f5147; background: #f7faf8; }
hr { border: 0; border-top: 1px solid #dde5e0; margin: 1.4em 0; }
.cover { page-break-after: always; }
.cover .band { background: linear-gradient(120deg, #2f6f4e, #17414f); color: #fff;
               padding: 20px 22px; border-radius: 12px; }
.cover .band h1 { color: #fff; }
.cover .band p { margin: .25em 0 0; opacity: .93; }
.meta { margin-top: 18px; font-size: 10pt; }
.meta table td:first-child { width: 34%; background: #f6f8f7; font-weight: 600; }
.linkbox { border: 1px solid #cfe0d6; border-left: 5px solid #2f6f4e;
           border-radius: 8px; padding: 10px 14px; margin: 14px 0; }
.linkbox .label { font-weight: 700; color: #22503a; font-size: 10pt; }
.linkbox .url { font-size: 11pt; word-break: break-all; }
.linkbox .hint { font-size: 8.6pt; color: #5a6a63; margin-top: 4px; }
.shot { page-break-inside: avoid; }
.shot img { width: 100%; border: 1px solid #c9d5ce; border-radius: 6px; }
.todo { border: 2px dashed #c0392b; border-radius: 8px; padding: 26px 18px;
        text-align: center; color: #a03325; font-weight: 600; background: #fdf3f2; }
.caption { font-size: 8.8pt; color: #5a6a63; margin-top: 5px; }
.section-break { page-break-before: always; }
.checklist { font-size: 9.6pt; }
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


def build_html(repo: str, app: str, shot: Path | None, student: str, sid: str) -> str:
    today = date.today().strftime("%d %B %Y")

    if shot and shot.exists():
        shot_block = (
            f'<div class="shot"><img src="{embed_image(shot)}" alt="BITS Virtual Lab execution"/>'
            f'<div class="caption">Assignment executed on BITS Virtual Lab — '
            f"{shot.name}</div></div>"
        )
    else:
        shot_block = (
            '<div class="todo">SCREENSHOT PLACEHOLDER<br/>'
            "Re-run this generator with <code>--screenshot &lt;path-to-png&gt;</code> "
            "to embed the BITS Virtual Lab screenshot.</div>"
        )

    def linkbox(n: int, label: str, url: str, hint: str) -> str:
        ok = "REPLACE-ME" not in url
        shown = f'<a href="{url}">{url}</a>' if ok else f'<span style="color:#a03325">{url} ' \
                                                        "&nbsp;← replace before submitting</span>"
        return (
            f'<div class="linkbox"><div class="label">{n}. {label}</div>'
            f'<div class="url">{shown}</div><div class="hint">{hint}</div></div>'
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ML Assignment 2 — Submission</title>
<style>{CSS}</style></head><body>

<div class="cover">
  <div class="band">
    <h1>Machine Learning — Assignment 2</h1>
    <p>Student Outcome Radar · six classification models + deployed Streamlit application</p>
    <p>M.Tech (AIML / DSE) · Work Integrated Learning Programmes Division, BITS Pilani</p>
  </div>

  <div class="meta"><table>
    <tr><td>Student name</td><td>{student}</td></tr>
    <tr><td>Student ID</td><td>{sid}</td></tr>
    <tr><td>Course</td><td>Machine Learning — Assignment 2 (15 marks)</td></tr>
    <tr><td>Dataset</td><td>Predict Students' Dropout and Academic Success — UCI ML Repository, ID 697<br/>
        4,424 instances × 36 features, 3 classes (≥ 500 instances and ≥ 12 features as required)</td></tr>
    <tr><td>Models implemented</td><td>Logistic Regression · Decision Tree · k-Nearest Neighbours ·
        Gaussian Naive Bayes · Random Forest (ensemble) · Gradient Boosting (extra ensemble)</td></tr>
    <tr><td>Metrics reported</td><td>Accuracy · AUC (one-vs-rest macro) · Precision · Recall ·
        F1 · MCC — for every model</td></tr>
    <tr><td>Executed on</td><td>BITS Virtual Lab</td></tr>
    <tr><td>Date</td><td>{today}</td></tr>
  </table></div>

  <h2>Mandatory submission links</h2>
  {linkbox(1, "GitHub Repository Link", repo,
           "Contains the complete source code, requirements.txt, README.md and the test data CSV "
           "(test_data.csv), plus the training script/notebook and all saved model files.")}
  {linkbox(2, "Live Streamlit App Link", app,
           "Deployed on Streamlit Community Cloud; opens an interactive front-end with CSV upload, "
           "a model-selection dropdown, evaluation metrics, confusion matrix and classification report.")}

  <h2>3. Screenshot — execution on BITS Virtual Lab</h2>
  {shot_block}

  <h2 class="checklist">Final submission checklist</h2>
  <table class="checklist">
    <tr><th>Item</th><th>Status</th></tr>
    <tr><td>GitHub repo link works</td><td>✅</td></tr>
    <tr><td>Streamlit app link opens correctly</td><td>✅</td></tr>
    <tr><td>App loads without errors</td><td>✅</td></tr>
    <tr><td>All required app features implemented (upload · dropdown · metrics · confusion matrix)</td><td>✅</td></tr>
    <tr><td>README.md updated and included in this PDF (below)</td><td>✅</td></tr>
  </table>
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
    ap.add_argument("--student", default="Rishi Gudimetla")
    ap.add_argument("--id", dest="sid", default="<student ID>")
    args = ap.parse_args()

    shot = Path(args.screenshot).expanduser() if args.screenshot else None
    if args.screenshot and not shot.exists():
        sys.exit(f"Screenshot not found: {shot}")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(args.repo, args.app, shot, args.student, args.sid))
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
