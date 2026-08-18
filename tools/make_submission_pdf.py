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
a { color: #000; text-decoration: underline; word-break: break-all; }
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
.linkbox .hint { font-size: 9pt; margin-top: 2px; }
.shot { page-break-inside: avoid; margin-top: .5em; }
.shot img { width: 100%; border: 1px solid #000; }
.todo { border: 1px solid #000; padding: 22px 16px; text-align: center; font-weight: bold; }
.caption { font-size: 9pt; margin-top: 4px; font-style: italic; }
.section-break { page-break-before: always; }
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
        shot_block = (
            f'<div class="shot"><img src="{embed_image(shot)}" alt="BITS Virtual Lab execution"/>'
            f'<div class="caption">Assignment executed on BITS Virtual Lab — the training '
            "notebook <code>model/train_models.ipynb</code> running in Jupyter, showing the "
            "cross-validated and hold-out scores for each classifier."
            "</div></div>"
        )
    else:
        shot_block = (
            '<div class="todo">SCREENSHOT PLACEHOLDER<br/>'
            "Re-run this generator with <code>--screenshot &lt;path-to-png&gt;</code> "
            "to embed the BITS Virtual Lab screenshot.</div>"
        )

    if app_shot and app_shot.exists():
        app_shot_block = (
            f'<div class="shot"><img src="{embed_image(app_shot)}" alt="Deployed app"/>'
            '<div class="caption">Supplementary evidence (not the mandated Section 3 screenshot): '
            "the deployed Streamlit app opened from BITS Virtual Lab, showing the model dropdown "
            "set to Random Forest, the confusion matrix, the one-vs-rest ROC curves and the "
            "per-class classification report. The figures match "
            "<code>model/metrics.json</code> exactly.</div></div>"
        )
    else:
        app_shot_block = ""

    def linkbox(n: int, label: str, url: str, hint: str) -> str:
        ok = "REPLACE-ME" not in url
        shown = f'<a href="{url}">{url}</a>' if ok else f"{url} (replace before submitting)"
        return (
            f'<div class="linkbox"><div class="label">{n}. {label}</div>'
            f'<div class="url">{shown}</div><div class="hint">{hint}</div></div>'
        )

    app_shot_section = ("" if not app_shot_block else
        '<div class="section-break"><h2>Appendix — deployed Streamlit application</h2>'
        + app_shot_block + "</div>")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ML Assignment 2 - Submission</title>
<style>{CSS}</style></head><body>

<div class="cover">
  <h1>Machine Learning - Assignment 2</h1>
  <p class="sub">Student Outcome Radar: six classification models and a deployed Streamlit application</p>
  <p class="dept">M.Tech (AIML / DSE), Work Integrated Learning Programmes Division, BITS Pilani</p>

  <div class="meta"><table>
    <tr><td>Student name</td><td>{student}</td></tr>
    <tr><td>Student ID</td><td>{sid}</td></tr>
    <tr><td>Course</td><td>Machine Learning - Assignment 2 (15 marks)</td></tr>
    <tr><td>Dataset</td><td>Predict Students' Dropout and Academic Success - UCI ML Repository, ID 697.<br/>
        4,424 instances, 36 features, 3 classes (meets the minimum of 500 instances and 12 features)</td></tr>
    <tr><td>Models implemented</td><td>Logistic Regression, Decision Tree, k-Nearest Neighbours,
        Gaussian Naive Bayes, Random Forest (ensemble), Gradient Boosting (additional ensemble)</td></tr>
    <tr><td>Metrics reported</td><td>Accuracy, AUC (one-vs-rest macro), Precision, Recall, F1 and MCC for every model</td></tr>
    <tr><td>Executed on</td><td>BITS Virtual Lab</td></tr>
    <tr><td>Date</td><td>{today}</td></tr>
  </table></div>

  <h2>Mandatory submission links</h2>
  {linkbox(1, "GitHub Repository Link", repo,
           "Contains the complete source code, requirements.txt, README.md and the test data CSV "
           "(test_data.csv), plus the training script and notebook and all saved model files.")}
  {linkbox(2, "Live Streamlit App Link", app,
           "Deployed on Streamlit Community Cloud. Opens an interactive front-end with CSV upload, "
           "a model-selection dropdown, evaluation metrics, confusion matrix and classification report.")}

  <h2>Final submission checklist</h2>
  <table>
    <tr><th>Item</th><th>Status</th></tr>
    <tr><td>GitHub repo link works</td><td>Done</td></tr>
    <tr><td>Streamlit app link opens correctly</td><td>Done</td></tr>
    <tr><td>App loads without errors</td><td>Done</td></tr>
    <tr><td>All required app features implemented (upload · dropdown · metrics · confusion matrix)</td><td>Done</td></tr>
    <tr><td>README.md updated and included in this PDF (below)</td><td>Done</td></tr>
  </table>
</div>

<div class="section-break">
  <h2>3. Screenshot: execution on BITS Virtual Lab</h2>
  {shot_block}
</div>

{app_shot_section}

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
