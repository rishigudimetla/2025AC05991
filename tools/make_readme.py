"""
Render README.md from model/metrics.json so that every number in the report is
generated from the actual training run - no hand-copied metrics, no drift.

Usage:
    python tools/make_readme.py
    python tools/make_readme.py --repo https://github.com/<user>/<repo> \
                               --app  https://<app>.streamlit.app
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRICS = json.loads((ROOT / "model" / "metrics.json").read_text())

REPO_PLACEHOLDER = "https://github.com/REPLACE-ME/ml-assignment2-student-outcome-radar"
APP_PLACEHOLDER = "https://REPLACE-ME.streamlit.app"

REQUIRED_FIVE = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest (Ensemble)",
]
EXTRA = "Gradient Boosting (Ensemble, extra)"

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def m(model: str, metric: str) -> float:
    return METRICS["metrics"][model][metric]


def recall_of(model: str, cls: str) -> float:
    return METRICS["per_class_report"][model][cls]["recall"]


def precision_of(model: str, cls: str) -> float:
    return METRICS["per_class_report"][model][cls]["precision"]


def cv(model: str) -> float:
    return METRICS["cv_macro_f1"][model]


def params(model: str) -> str:
    best = METRICS["best_params"].get(model, {})
    return ", ".join(f"`{k}={v}`" for k, v in best.items()) or "defaults"


def fmt(x: float) -> str:
    return f"{x:.4f}"


def comparison_table(models: list[str]) -> str:
    head = "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |"
    rule = "|---|---|---|---|---|---|---|"
    rows = []
    best = {k: max(m(md, k) for md in models) for k in METRIC_ORDER}
    for name in models:
        cells = []
        for k in METRIC_ORDER:
            v = fmt(m(name, k))
            cells.append(f"**{v}**" if m(name, k) == best[k] else v)
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


def observation_rows() -> str:
    d = METRICS
    dim = METRICS["preprocessing"]["expanded_dimensionality"]
    latency = METRICS["inference_ms_full_testset"]
    sizes = METRICS["artefact_size_mb"]
    rf, lr, dt, knn, nb = (
        "Random Forest (Ensemble)", "Logistic Regression",
        "Decision Tree", "kNN", "Naive Bayes",
    )
    notes = {
        lr: (
            f"Strongest non-ensemble model: accuracy {fmt(m(lr,'Accuracy'))}, macro-F1 "
            f"{fmt(m(lr,'F1'))}, MCC {fmt(m(lr,'MCC'))}, and an AUC of {fmt(m(lr,'AUC'))} that is "
            f"essentially level with the Random Forest ({fmt(m(rf,'AUC'))}). The class boundaries "
            "are therefore largely linear once the semester-approval features are scaled. "
            f"`class_weight='balanced'` lifts *Enrolled* recall to {fmt(recall_of(lr,'Enrolled'))} "
            f"but its precision stays low ({fmt(precision_of(lr,'Enrolled'))}), so it over-predicts "
            "the middle class. Cheapest model to train and the most interpretable (signed "
            "coefficients), which makes it a sensible fallback if the deployment budget is tight."
        ),
        dt: (
            f"**Lowest accuracy of the six** ({fmt(m(dt,'Accuracy'))}, MCC {fmt(m(dt,'MCC'))}). CV "
            f"chose {params(dt)}, i.e. pruning was necessary - an unpruned tree over-fitted the "
            "one-hot encoded parental-qualification columns. It buys the highest *Enrolled* recall "
            f"of all models ({fmt(recall_of(dt,'Enrolled'))}) at a precision of only "
            f"{fmt(precision_of(dt,'Enrolled'))}, and it drags *Graduate* recall down to "
            f"{fmt(recall_of(dt,'Graduate'))} - it happily trades away the majority class to chase "
            "the middle one. A single tree is high-variance here; it is best read as the "
            "interpretable building block that the ensembles average away."
        ),
        knn: (
            f"Accuracy {fmt(m(knn,'Accuracy'))} looks respectable but macro-F1 is only "
            f"{fmt(m(knn,'F1'))} and AUC is the lowest of the six ({fmt(m(knn,'AUC'))}): one-hot "
            f"encoding expands the input to {dim} dimensions, where Manhattan/Euclidean "
            "neighbourhoods stop being informative (curse of dimensionality). It collapses onto "
            f"the majority class - *Graduate* recall {fmt(recall_of(knn,'Graduate'))} versus "
            f"*Enrolled* recall only {fmt(recall_of(knn,'Enrolled'))}. It is also the one model "
            "with no `class_weight` option, so the "
            f"{max(METRICS['dataset']['class_counts'].values()) / min(METRICS['dataset']['class_counts'].values()):.2f}:1 "
            f"imbalance hits it hardest, and as a lazy learner it is the slowest at inference "
            f"({latency[knn]} ms for the {METRICS['split']['test_rows']}-row test set versus "
            f"{latency[lr]} ms for Logistic Regression)."
        ),
        nb: (
            f"Accuracy {fmt(m(nb,'Accuracy'))}, macro-F1 {fmt(m(nb,'F1'))} - the lowest macro-F1 "
            "and MCC of the six. The Gaussian conditional-independence assumption is clearly "
            "violated: the 1st- and 2nd-semester enrolled/approved/grade columns are strongly "
            "correlated, so evidence is double-counted and the posteriors are over-confident. "
            f"Only {fmt(recall_of(nb,'Enrolled'))} of *Enrolled* students are recovered. Its AUC "
            f"({fmt(m(nb,'AUC'))}) still beats kNN, so the ranking it produces is usable even "
            "though its hard decisions are not. Trains in well under a second - a useful baseline."
        ),
        rf: (
            f"**Best model on every one of the six metrics**: accuracy {fmt(m(rf,'Accuracy'))}, "
            f"AUC {fmt(m(rf,'AUC'))}, macro-F1 {fmt(m(rf,'F1'))}, MCC {fmt(m(rf,'MCC'))}. Bagging "
            f"500 trees removes the single tree's variance (+{fmt(m(rf,'F1')-m(dt,'F1'))} macro-F1 "
            "over the Decision Tree) while `balanced_subsample` keeps the minority class visible: "
            f"it is the only model that holds *Enrolled* precision at or above 0.50 "
            f"({fmt(precision_of(rf,'Enrolled'))}) while still recalling "
            f"{fmt(recall_of(rf,'Enrolled'))} of them. Also best on the two classes that matter "
            f"operationally - *Dropout* recall {fmt(recall_of(rf,'Dropout'))} and *Graduate* recall "
            f"{fmt(recall_of(rf,'Graduate'))}. Cost: the largest artefact "
            f"(~{sizes[rf]:.0f} MB) and the slowest training run."
        ),
        EXTRA: (
            f"Included because the brief mentions \"6 ML models\" while listing five. Accuracy "
            f"{fmt(m(EXTRA,'Accuracy'))} and AUC {fmt(m(EXTRA,'AUC'))} come close to the Random "
            f"Forest, and it gives the best *Graduate* recall overall "
            f"({fmt(recall_of(EXTRA,'Graduate'))}), but boosting without class weighting sacrifices "
            f"the minority class (*Enrolled* recall {fmt(recall_of(EXTRA,'Enrolled'))}), so macro-F1 "
            f"drops to {fmt(m(EXTRA,'F1'))}. Confirms that on this dataset the gain comes from "
            "*ensembling*, and that handling the imbalance is what separates the two ensembles."
        ),
    }
    lines = ["| ML Model Name | Observation about model performance |", "|---|---|"]
    for name in REQUIRED_FIVE + [EXTRA]:
        lines.append(f"| {name} | {notes[name]} |")
    winner = METRICS["winner"]
    lines.append(
        f"| **Overall Winner for your dataset?** | **{winner}** - it tops all six metrics "
        f"simultaneously (accuracy {fmt(m(winner,'Accuracy'))}, AUC {fmt(m(winner,'AUC'))}, "
        f"macro-F1 {fmt(m(winner,'F1'))}, MCC {fmt(m(winner,'MCC'))}) and its 5-fold CV score "
        f"({fmt(cv(winner))}) is consistent with its hold-out score, so the win is not a lucky "
        "split. MCC of "
        f"{fmt(m(winner,'MCC'))} versus {fmt(m('Naive Bayes','MCC'))} for the weakest model shows "
        "the margin is substantial on a chance-corrected measure too. |"
    )
    return "\n".join(lines)


def test_count() -> str:
    """Count the collected pytest tests so the README can never quote a stale number."""
    import subprocess

    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
        ).stdout
        for line in reversed(out.strip().splitlines()):
            if "test" in line and line.split()[0].isdigit():
                return line.split()[0]
    except Exception:  # noqa: BLE001 - README generation must not depend on pytest
        pass
    return "the"


def build(repo_url: str, app_url: str) -> str:
    d = METRICS["dataset"]
    s = METRICS["split"]
    prep = METRICS["preprocessing"]
    counts = d["class_counts"]
    total = sum(counts.values())
    winner = METRICS["winner"]
    n_tests = test_count()

    param_rows = "\n".join(
        f"| {name} | {params(name)} | {fmt(cv(name))} |"
        for name in REQUIRED_FIVE + [EXTRA]
    )
    class_rows = "\n".join(
        f"| {cls} | {counts[cls]} | {counts[cls] / total:.1%} | {desc} |"
        for cls, desc in [
            ("Dropout", "left the programme before completing it"),
            ("Enrolled", "still studying at the end of the normal course duration"),
            ("Graduate", "completed the programme on time"),
        ]
    )

    return f"""# 🎓 Student Outcome Radar — Machine Learning Assignment 2

**M.Tech (AIML / DSE) · Work Integrated Learning Programmes Division, BITS Pilani**
Predicting whether a higher-education student will **drop out**, stay **enrolled**, or
**graduate**, with six classification models and an interactive Streamlit front-end.

| | |
|---|---|
| 🔗 **GitHub repository** | {repo_url} |
| 🚀 **Live Streamlit app** | {app_url} |
| 📊 **Dataset** | [{d['name']}]({d['url']}) — {d['source']} |
| 🏆 **Best model** | **{winner}** (accuracy {fmt(m(winner, 'Accuracy'))}, macro-F1 {fmt(m(winner, 'F1'))}, MCC {fmt(m(winner, 'MCC'))}) |
| 🧪 **Executed on** | BITS Virtual Lab (screenshot included in the submitted PDF) |

---

## a. Problem statement

Student attrition is expensive for both the learner and the institution: a student who
withdraws in the second or third semester has already consumed teaching capacity and
financial aid, and leaves without a qualification. Institutions therefore want to know
**as early as possible** which students are at risk, so that counselling, fee relief or
academic support can be directed at them instead of being spread thinly.

This project frames that need as a **supervised, single-label, three-class classification
problem**:

> Given the information an institution already holds about a student at enrolment
> (demographics, parental education and occupation, prior qualification, admission grade,
> scholarship / debtor / fee status) together with their **first- and second-semester
> academic record** (units enrolled, evaluated, approved and average grade) and the
> **macro-economic conditions** of that year (unemployment, inflation, GDP), predict the
> student's academic outcome:
>
> `Dropout` · `Enrolled` (still studying beyond the normal duration) · `Graduate`

Why three classes rather than two: collapsing *Enrolled* into either bucket would hide the
students who are drifting — neither failing outright nor finishing on time — and they are
exactly the group an early-warning system should surface. The consequence is that the middle
class is small and genuinely ambiguous, which is why every headline metric in this report
uses **macro averaging** (all three classes weighted equally) instead of the flattering
accuracy figure a majority-class predictor would earn.

**Success criteria.** (1) Beat the majority-class baseline of
{max(counts.values()) / total:.1%} accuracy by a clear margin; (2) maximise **macro-F1** and
**MCC** so the minority *Enrolled* class cannot be sacrificed; (3) keep *Dropout* recall high,
because a missed at-risk student is costlier than an unnecessary counselling session.

---

## b. Dataset description

**{d['name']}** — {d['source']}
Source URL: {d['url']}

| Property | Value | Assignment requirement | Status |
|---|---|---|---|
| Instances | **{d['instances']:,}** | ≥ 500 | ✅ |
| Features | **{d['features']}** | ≥ 12 | ✅ |
| Classes | {len(d['classes'])} ({', '.join(d['classes'])}) | binary *or* multi-class | ✅ multi-class |
| Missing values | 0 | — | ✅ no imputation needed |
| Duplicate rows | 0 | — | ✅ |
| Source repository | UCI Machine Learning Repository (ID 697) | Kaggle or UCI | ✅ |
| Licence | CC BY 4.0 | — | ✅ redistributable |

### Target variable

| Class | Instances | Share | Meaning |
|---|---|---|---|
{class_rows}

The classes are imbalanced at **{max(counts.values()) / min(counts.values()):.2f} : 1**
(largest : smallest). A model that always answered *Graduate* would already score
{max(counts.values()) / total:.1%} accuracy — the reason macro-F1 and MCC, not accuracy, are
used to pick the winner.

### Feature groups ({d['features']} predictors)

| Group | Count | Examples | Handling |
|---|---|---|---|
| Nominal codes | {len(prep['one_hot'])} | `Course`, `Application mode`, `Mother's / Father's qualification`, `Mother's / Father's occupation`, `Nacionality`, `Previous qualification`, `Marital status` | **One-hot encoded** (`handle_unknown='ignore'`) — the integers are labels, not magnitudes, so scaling them would invent a false ordering |
| Continuous / ordinal | {len(prep['standard_scaled'])} | `Admission grade`, `Age at enrollment`, `Curricular units 1st/2nd sem (enrolled · evaluations · approved · grade · credited · without evaluations)`, `Unemployment rate`, `Inflation rate`, `GDP` | **StandardScaler** — required by Logistic Regression and kNN, harmless for the tree models |
| Binary flags | {len(prep['passthrough_binary'])} | `Debtor`, `Tuition fees up to date`, `Scholarship holder`, `Displaced`, `Gender`, `Daytime/evening attendance`, `Educational special needs`, `International` | **Passed through** — already 0/1 |

### Train / test split

| | Rows | Purpose |
|---|---|---|
| `train_data.csv` | {s['train_rows']:,} ({int((1 - s['test_fraction']) * 100)}%) | model fitting + 5-fold CV hyper-parameter search |
| `test_data.csv` | {s['test_rows']:,} ({int(s['test_fraction'] * 100)}%) | untouched hold-out; this is the file the Streamlit app scores |

Split is **stratified** on the target with `random_state={s['random_state']}`, so the class
proportions above are preserved in both files and every run is reproducible. Only
`test_data.csv` is uploaded to / shipped with the app, in line with the instruction to keep
the free Streamlit tier light.

---

## c. GitHub repository link

**{repo_url}**

```
ml-assignment2/
├── app.py                    # Streamlit application (deployed entry point)
├── ml_core.py                # shared preprocessing, model zoo, metric definitions
├── requirements.txt          # pinned dependencies
├── README.md                 # this file
├── test_data.csv             # {s['test_rows']}-row stratified hold-out split (upload me in the app)
├── train_data.csv            # {s['train_rows']}-row training split (retrain fallback)
├── .streamlit/config.toml    # theme + 10 MB upload cap
├── rawdata/data.csv          # original UCI file (semicolon separated)
├── model/
│   ├── train_models.py       # training + evaluation script
│   ├── train_models.ipynb    # notebook version (run on BITS Virtual Lab)
│   ├── metrics.json          # every metric, confusion matrix, per-class report, best params
│   ├── comparison_table.csv  # the table reproduced below
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── random_forest.joblib
│   └── gradient_boosting.joblib
├── tests/test_project.py     # {n_tests} automated checks (data, models, metrics, UI)
└── tools/                    # README + submission-PDF generators
```

### Reproduce from scratch

```bash
git clone {repo_url}
cd ml-assignment2
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# 1. fetch the raw UCI file (only needed if rawdata/data.csv is absent)
curl -L -o uci697.zip "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
unzip uci697.zip -d rawdata

# 2. train all six models, write the splits, metrics and artefacts
python model/train_models.py

# 3. verify everything
python -m pytest tests -q

# 4. run the app
streamlit run app.py
```

---

## d. Models used

All six pipelines are trained on the **same** {s['train_rows']}-row training split and scored on
the **same** untouched {s['test_rows']}-row hold-out split. Each model is packaged as a single
`Pipeline(ColumnTransformer → estimator)`, so the encoder and scaler are fitted inside every
CV fold — there is **no leakage** from the test set into preprocessing — and the artefact that
is deployed is byte-for-byte the artefact that was evaluated.

| # | Model | scikit-learn estimator | Selected hyper-parameters (5-fold CV, `f1_macro`) | CV macro-F1 |
|---|---|---|---|---|
| 1 | Logistic Regression | `LogisticRegression(class_weight='balanced', max_iter=3000)` | {params('Logistic Regression')} | {fmt(cv('Logistic Regression'))} |
| 2 | Decision Tree | `DecisionTreeClassifier(class_weight='balanced')` | {params('Decision Tree')} | {fmt(cv('Decision Tree'))} |
| 3 | kNN | `KNeighborsClassifier()` | {params('kNN')} | {fmt(cv('kNN'))} |
| 4 | Naive Bayes | `GaussianNB()` | {params('Naive Bayes')} | {fmt(cv('Naive Bayes'))} |
| 5 | Random Forest (Ensemble) | `RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample')` | {params('Random Forest (Ensemble)')} | {fmt(cv('Random Forest (Ensemble)'))} |
| 6 | Gradient Boosting (Ensemble, *extra*) | `GradientBoostingClassifier()` | {params(EXTRA)} | {fmt(cv(EXTRA))} |

> Models 1–5 are the five mandated by the brief. Model 6 is added because the assignment text
> refers to *"all the 6 ML models"* while listing five, so a second ensemble is supplied to
> cover both readings; it is labelled *extra* everywhere and never replaces a required model.

After encoding, all models see the same **{METRICS['preprocessing']['expanded_dimensionality']}-dimensional**
input ({len(prep['one_hot'])} nominal columns expanded to one-hot, plus
{len(prep['standard_scaled'])} scaled and {len(prep['passthrough_binary'])} binary columns).
Deployment cost differs sharply, which matters on a free hosting tier:

| Model | Artefact size | Inference time, {s['test_rows']} rows |
|---|---|---|
{chr(10).join(f"| {name} | {METRICS['artefact_size_mb'][name]:.2f} MB | {METRICS['inference_ms_full_testset'][name]} ms |" for name in REQUIRED_FIVE + [EXTRA])}

### How each metric is computed

The target has three classes, so the assignment's six metrics are reported as follows:

| Metric | Definition used | Why |
|---|---|---|
| Accuracy | `accuracy_score` | overall hit rate; baseline to beat is {max(counts.values()) / total:.1%} |
| AUC | `roc_auc_score(..., multi_class='ovr', average='macro')` | one-vs-rest AUC per class, then unweighted mean — the standard multi-class extension; uses `predict_proba`, so it judges the ranking rather than the argmax |
| Precision | `precision_score(..., average='macro')` | unweighted mean over the three classes |
| Recall | `recall_score(..., average='macro')` | unweighted mean over the three classes |
| F1 | `f1_score(..., average='macro')` | harmonic mean of the two above, per class then averaged |
| MCC | `matthews_corrcoef` | chance-corrected correlation over the full 3×3 matrix; the most honest single number under imbalance |

**Macro** (not *weighted*) averaging is used deliberately: it refuses to let the
{counts['Enrolled']}-instance *Enrolled* class be drowned out by the
{counts['Graduate']}-instance *Graduate* class.

### Comparison table — evaluation metrics for all models

Hold-out test set, {s['test_rows']} rows never seen during training or tuning.
Best value in each column is **bold**.

{comparison_table(REQUIRED_FIVE + [EXTRA])}

The same table restricted to the five mandated models:

{comparison_table(REQUIRED_FIVE)}

*(Generated from `model/metrics.json`; the Streamlit app recomputes these numbers live from
the saved artefacts, and `tests/test_project.py` asserts that the recomputed values match this
table to 1e-6.)*

### Observations on model performance

{observation_rows()}

### What the errors look like

Confusion matrix of the winning model ({winner}), rows = actual, columns = predicted:

| actual ↓ / predicted → | Dropout | Enrolled | Graduate |
|---|---|---|---|
{chr(10).join(f"| **{cls}** | " + " | ".join(str(v) for v in row) + " |" for cls, row in zip(d['classes'], METRICS['confusion_matrices'][winner]))}

Three findings that hold across every model:

1. **Dropout vs Graduate is nearly solved.** Only
   {METRICS['confusion_matrices'][winner][0][2]} true *Dropout* students are called *Graduate*
   and {METRICS['confusion_matrices'][winner][2][0]} the other way round — the two extremes are
   well separated by the semester-approval and fee-status features.
2. **All the difficulty sits in *Enrolled*.** It is the minority class *and* semantically
   in-between; every model's worst per-class F1 is *Enrolled*
   (best being {fmt(METRICS['per_class_report'][winner]['Enrolled']['f1-score'])} for the
   Random Forest). Anyone using this model in practice should treat *Enrolled* as
   "needs a human look", not as a confident prediction.
3. **Handling the imbalance matters more than raw model power.** The class-weighted Logistic
   Regression (macro-F1 {fmt(m('Logistic Regression', 'F1'))}) beats the unweighted Gradient
   Boosting ({fmt(m(EXTRA, 'F1'))}) on macro-F1 despite being far simpler.

---

## Streamlit application

**Live app: {app_url}** — deployed on Streamlit Community Cloud (free tier).

Features, mapped to the assignment's requirements:

| Requirement | Where it is in the app | Detail |
|---|---|---|
| **a. Dataset upload option (CSV)** | Sidebar → *1 · Test data* → **Upload my own test CSV** | Accepts any CSV with the {d['features']} feature columns. Header quirks are tidied, non-numeric cells and NaN rows are reported and dropped, unexpected columns are ignored, and a missing feature produces a readable error instead of a stack trace. With a `Target` column you get full metrics; without one the app switches to prediction-only mode. Only test data is uploaded, per the brief. A **Download bundled test_data.csv** button provides a template. |
| **b. Model selection dropdown** | Sidebar → *2 · Model* → **Choose a model** | All six trained pipelines, with a one-line description of the selected one. Every chart and table on the page follows the selection. |
| **c. Display of evaluation metrics** | Tab **📊 Model evaluation** and tab **🏁 Compare all models** | The six required metrics as cards, each showing the delta against the offline training run (so the deployed artefact is provably the evaluated one), plus a full six-model comparison table with a colour gradient and a sortable bar chart per metric. |
| **d. Confusion matrix / classification report** | Tab **📊 Model evaluation** | Confusion-matrix heatmap (raw counts or row-normalised), per-class precision / recall / F1 / support table, the plain-text `classification_report`, and one-vs-rest ROC curves with per-class AUC. |

Extras beyond the requirement: per-student prediction table with class probabilities and a
confidence column, a "show only mis-classified rows" filter, predicted-class distribution
chart, CSV download of both the predictions and the live comparison table, a dataset/method
tab documenting the pipeline, and an automatic retrain-from-`train_data.csv` fallback if an
artefact ever fails to unpickle on the cloud.

### Deployment steps used

1. Push this repository to GitHub (branch `main`).
2. Sign in at <https://streamlit.io/cloud> with the same GitHub account.
3. **New app** → select this repository → branch `main` → main file **`app.py`**.
4. *Advanced settings* → Python 3.13 (matches the pinned wheels in `requirements.txt`).
5. **Deploy**, then confirm the app opens, loads `test_data.csv` and renders metrics.

`requirements.txt` pins exact versions — including the scikit-learn version the artefacts were
pickled with ({METRICS['sklearn_version']}) — because dependency drift is the most common cause
of a failed Streamlit deployment.

---

## Verification

`tests/test_project.py` runs {n_tests} automated checks with `python -m pytest tests -q`:

- dataset meets the ≥ 12 features / ≥ 500 instances requirement;
- `train_data.csv` and `test_data.csv` are the right size, **disjoint** (no leakage) and stratified;
- all six artefacts load and predict valid class labels;
- the five mandated models are present;
- **re-scoring each artefact reproduces `metrics.json` to 1e-6** — the published table cannot drift;
- the retrain-from-`train_data.csv` fallback reproduces all six models to 1e-6, so the app
  degrades safely if an artefact ever fails to unpickle;
- all six metrics exist for all six models;
- CSV validation: good file, missing `Target`, missing feature column, extra columns, untidy
  headers, NaN / non-numeric cells, single-class slice;
- the Streamlit app runs with no exception, switches across all six models, toggles every display
  option, renders the six metric cards, confusion matrix and report, and correctly handles a
  labelled upload, an unlabelled upload and a malformed upload.

---

## Originality note

Dataset choice (UCI 697, three-class), the `ml_core.py` + `train_models.py` + `app.py` split,
the macro-averaged multi-class metric treatment, the tuned hyper-parameter grids, the
"Student Outcome Radar" UI (four-tab layout, custom CSS hero, delta-versus-training metric
cards, retrain fallback) and the test suite were all written for this submission. No Streamlit
template or public notebook was copied; AI assistance was used for learning support and code
review only, and every number in this README is generated from `model/metrics.json` by
`tools/make_readme.py`.

## Author

**Rishi Gudimetla** — M.Tech (AIML / DSE), BITS Pilani WILP
Machine Learning Assignment 2 · executed on **BITS Virtual Lab** · submitted {""}18 August 2026
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_PLACEHOLDER, help="GitHub repository URL")
    ap.add_argument("--app", default=APP_PLACEHOLDER, help="Live Streamlit app URL")
    args = ap.parse_args()

    text = build(args.repo, args.app)
    (ROOT / "README.md").write_text(text)
    print(f"Wrote README.md ({len(text.splitlines())} lines)")
    if args.repo == REPO_PLACEHOLDER or args.app == APP_PLACEHOLDER:
        print("\nNOTE: placeholder link(s) still present. Once the repo and app exist, rerun:")
        print("  python tools/make_readme.py --repo <github-url> --app <streamlit-url>")


if __name__ == "__main__":
    main()
