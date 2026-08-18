"""
=====================================================================
 Student Outcome Radar  -  ML Assignment 2 Streamlit application
 M.Tech (AIML / DSE), WILP Division
---------------------------------------------------------------------
 Interactive front-end for six pre-trained classification pipelines
 built on the UCI dataset "Predict Students' Dropout and Academic
 Success" (ID 697).

 Assignment-mandated features
   a. Test-data upload (CSV)              -> sidebar "Upload test CSV"
   b. Model selection dropdown            -> sidebar "Choose a model"
   c. Display of evaluation metrics       -> tab 1 + tab 2
   d. Confusion matrix / class. report    -> tab 1

 Local run:  streamlit run app.py
=====================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import auc as auc_of_curve
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

from ml_core import (
    METRIC_ORDER,
    METRICS_JSON,
    MODEL_DIR,
    MODEL_NOTES,
    TARGET,
    TEST_CSV,
    TRAIN_CSV,
    build_pipeline,
    evaluate,
    metrics_to_frame,
    prepare_test_frame,
    tidy_frame,
)

# --------------------------------------------------------------------
# Page shell
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Student Outcome Radar | ML Assignment 2",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#2f6f4e"
PALETTE = {"Dropout": "#c0392b", "Enrolled": "#d68910", "Graduate": "#1e8449"}

st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 1.6rem; padding-bottom: 2.5rem;}}
      .hero {{
        background: linear-gradient(120deg, {ACCENT} 0%, #17414f 100%);
        color: #ffffff; padding: 1.5rem 1.8rem; border-radius: 14px;
        margin-bottom: 1.4rem;
      }}
      .hero h1 {{margin: 0 0 .35rem 0; font-size: 1.85rem; font-weight: 700;}}
      .hero p  {{margin: 0; opacity: .92; font-size: .96rem; line-height: 1.5;}}
      .pill {{
        display:inline-block; background: rgba(255,255,255,.18);
        border: 1px solid rgba(255,255,255,.35); border-radius: 999px;
        padding: .12rem .7rem; margin: .45rem .35rem 0 0; font-size: .78rem;
      }}
      div[data-testid="stMetric"] {{
        background:#f6f8f7; border:1px solid #e0e6e3;
        border-radius:12px; padding:.7rem .9rem;
      }}
      div[data-testid="stMetricLabel"] p {{font-weight:600; color:#44544c;}}
      .note {{font-size:.85rem; color:#5a6a63;}}
      footer {{visibility:hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if not METRICS_JSON.exists():
        st.error("model/metrics.json is missing. Run `python model/train_models.py` first.")
        st.stop()
    return json.loads(METRICS_JSON.read_text())


@st.cache_resource(show_spinner="Loading trained pipelines…")
def load_models(model_files: tuple[tuple[str, str], ...]) -> tuple[dict, list[str]]:
    """Load every saved pipeline. Falls back to an in-app retrain if a
    pickle cannot be read (e.g. a scikit-learn version drift on the cloud)."""
    models, rebuilt = {}, []
    for name, filename in model_files:
        path = MODEL_DIR / filename
        try:
            models[name] = joblib.load(path)
        except Exception:  # noqa: BLE001 - any unpickling problem triggers the fallback
            models[name] = _retrain(name)
            rebuilt.append(name)
    return models, rebuilt


def _retrain(model_name: str):
    """Safety net: refit the pipeline from train_data.csv using the stored best params."""
    meta = load_metadata()
    train = tidy_frame(pd.read_csv(TRAIN_CSV))
    X, y = train.drop(columns=[TARGET]), train[TARGET]
    params = {f"{k}": v for k, v in meta["best_params"].get(model_name, {}).items()}
    pipe = build_pipeline(model_name, X, params)
    return pipe.fit(X, y)


@st.cache_data(show_spinner=False)
def load_bundled_test() -> pd.DataFrame:
    return tidy_frame(pd.read_csv(TEST_CSV))


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def prepare_frame(raw: pd.DataFrame, expected: list[str]):
    """Thin Streamlit wrapper around ml_core.prepare_test_frame."""
    X, y, notes, errors = prepare_test_frame(raw, expected)
    if errors:
        for message in errors:
            st.error(message)
        st.info("Tip: download the bundled `test_data.csv` from the sidebar as a template.")
        st.stop()
    return X, y, notes


def confusion_figure(matrix: list[list[int]], labels: list[str], normalise: bool):
    counts = np.array(matrix, dtype=int)
    if normalise:
        row_sums = counts.sum(axis=1, keepdims=True)
        data = np.divide(counts, row_sums, out=np.zeros(counts.shape, dtype=float),
                         where=row_sums != 0)
        fmt = ".2f"
    else:
        data, fmt = counts, "d"
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="YlGnBu",
        cbar=False,
        linewidths=0.6,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Row-normalised confusion matrix" if normalise else "Confusion matrix", pad=10)
    fig.tight_layout()
    return fig


def roc_figure(model, X: pd.DataFrame, y: pd.Series, labels: list[str]):
    if not hasattr(model, "predict_proba"):
        return None
    present = [c for c in labels if c in set(y)]
    if len(present) < 2:
        return None

    proba = model.predict_proba(X)
    order = list(model.classes_)
    y_bin = label_binarize(y, classes=order)
    if y_bin.shape[1] == 1:  # binary edge case
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(4.9, 3.8))
    for idx, cls in enumerate(order):
        if cls not in present:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, idx], proba[:, idx])
        ax.plot(fpr, tpr, lw=2, color=PALETTE.get(cls), label=f"{cls} (AUC {auc_of_curve(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves", pad=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def style_comparison(frame: pd.DataFrame):
    return (
        frame.style.format("{:.4f}", na_rep="n/a")
        .background_gradient(cmap="YlGn", axis=0)
        .set_properties(**{"text-align": "center"})
    )


# --------------------------------------------------------------------
# Load everything
# --------------------------------------------------------------------
meta = load_metadata()
FEATURES: list[str] = meta["feature_columns"]
CLASSES: list[str] = meta["dataset"]["classes"]
models, rebuilt = load_models(tuple(meta["models"].items()))
reference = metrics_to_frame(
    {k: {m: (np.nan if v is None else v) for m, v in vals.items()} for k, vals in meta["metrics"].items()}
)

st.markdown(
    f"""
    <div class="hero">
      <h1>🎓 Student Outcome Radar</h1>
      <p>Six classification pipelines predicting whether a higher-education student will
      <b>drop out</b>, remain <b>enrolled</b>, or <b>graduate</b>, from data known at enrolment
      plus first- and second-semester academic records.</p>
      <span class="pill">UCI dataset 697</span>
      <span class="pill">{meta['dataset']['instances']} instances</span>
      <span class="pill">{meta['dataset']['features']} features</span>
      <span class="pill">3 classes</span>
      <span class="pill">ML Assignment 2 &middot; M.Tech AIML/DSE</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if rebuilt:
    st.warning(
        "Saved artefacts for "
        + ", ".join(rebuilt)
        + " could not be unpickled in this environment, so they were refitted "
          "from `train_data.csv` using the stored best hyper-parameters."
    )

# --------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("1 · Test data")
    source = st.radio(
        "Evaluation data source",
        ["Bundled hold-out test set", "Upload my own test CSV"],
        key="data_source",
        help="The bundled file is the 20% stratified hold-out split never seen during training.",
    )

    uploaded = None
    if source == "Upload my own test CSV":
        uploaded = st.file_uploader("Upload test CSV", type=["csv"], key="csv_upload")
        st.caption(
            "Needs the 36 original feature columns; include the `Target` column to get metrics."
        )
        if uploaded is None:
            st.info("Waiting for a file — showing the bundled hold-out set meanwhile.")

    st.subheader("2 · Model")
    model_name = st.selectbox(
        "Choose a model", list(models.keys()), index=len(models) - 2, key="model_choice"
    )
    st.caption(MODEL_NOTES.get(model_name, ""))

    st.subheader("3 · Display options")
    normalise_cm = st.checkbox("Normalise confusion matrix by row", value=False, key="norm_cm")
    show_roc = st.checkbox("Show one-vs-rest ROC curves", value=True, key="show_roc")
    row_limit = st.slider("Rows to preview in tables", 5, 100, 15, step=5, key="row_limit")

    st.divider()
    with open(TEST_CSV, "rb") as handle:
        st.download_button(
            "⬇️ Download bundled test_data.csv",
            handle.read(),
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )
    st.caption(
        f"Trained with scikit-learn {meta.get('sklearn_version', 'n/a')} · "
        f"seed {meta['split']['random_state']} · {meta['averaging']}"
    )

# --------------------------------------------------------------------
# Assemble the evaluation frame
# --------------------------------------------------------------------
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    origin = f"uploaded file `{uploaded.name}`"
else:
    raw = load_bundled_test()
    origin = "bundled hold-out `test_data.csv`"

X, y, notes = prepare_frame(raw, FEATURES)
for note in notes:
    st.info(note)

st.caption(
    f"Evaluating **{len(X)}** rows from {origin} · "
    + (f"labels present ✅" if y is not None else "labels absent — prediction-only mode")
)

model = models[model_name]
scored = evaluate(model, X, y, CLASSES) if y is not None else None

tab_single, tab_compare, tab_rows, tab_about = st.tabs(
    ["📊 Model evaluation", "🏁 Compare all models", "🔎 Row-level predictions", "📚 Dataset & method"]
)

# ---------------------- TAB 1 : single model -----------------------
with tab_single:
    st.subheader(f"{model_name} — performance on the current test data")

    if scored is None:
        st.warning(
            "Ground-truth labels are required for evaluation metrics. "
            "Add a `Target` column to the CSV, or switch back to the bundled test set."
        )
    else:
        ref = reference.loc[model_name] if model_name in reference.index else None
        cols = st.columns(6)
        for col, metric in zip(cols, METRIC_ORDER):
            value = scored[metric]
            delta = None
            if ref is not None and not np.isnan(ref[metric]) and not np.isnan(value):
                diff = value - ref[metric]
                delta = f"{diff:+.4f} vs training run"
            col.metric(
                metric,
                "n/a" if np.isnan(value) else f"{value:.4f}",
                delta=delta,
                delta_color="normal" if delta else "off",
            )
        st.markdown(
            '<p class="note">Precision / Recall / F1 use <b>macro</b> averaging (every class weighted '
            'equally, so the small “Enrolled” class is not hidden). AUC is one-vs-rest macro. '
            "MCC is computed on the full 3×3 confusion matrix.</p>",
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)
        with left:
            st.pyplot(confusion_figure(scored["confusion_matrix"], CLASSES, normalise_cm))
        with right:
            if show_roc:
                fig = roc_figure(model, X, y, CLASSES)
                if fig is None:
                    st.info("ROC curves need `predict_proba` and at least two classes present.")
                else:
                    st.pyplot(fig)
            else:
                st.info("Enable “Show one-vs-rest ROC curves” in the sidebar.")

        st.markdown("**Classification report (per class)**")
        report = pd.DataFrame(scored["per_class"]).T
        report = report.loc[[c for c in CLASSES if c in report.index] +
                            [r for r in ("macro avg", "weighted avg") if r in report.index]]
        report["support"] = report["support"].astype(int)
        st.dataframe(
            report.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                 "f1-score": "{:.4f}", "support": "{:d}"}),
            width="stretch",
        )

        with st.expander("Plain-text classification report"):
            st.code(scored["report_text"], language="text")

        best_params = meta["best_params"].get(model_name, {})
        if best_params:
            st.markdown("**Tuned hyper-parameters (5-fold CV, macro-F1)**")
            st.json(best_params, expanded=False)

# ---------------------- TAB 2 : compare all ------------------------
with tab_compare:
    st.subheader("All six pipelines on the current test data")

    if y is None:
        st.warning("Upload a CSV that includes the `Target` column to compare models.")
    else:
        with st.spinner("Scoring every model…"):
            all_scores = {name: evaluate(mdl, X, y, CLASSES) for name, mdl in models.items()}
        table = metrics_to_frame({k: v for k, v in all_scores.items()})

        st.dataframe(style_comparison(table), width="stretch")

        champion = table["F1"].idxmax()
        c1, c2, c3 = st.columns(3)
        c1.metric("Best macro-F1", champion, f"{table.loc[champion, 'F1']:.4f}", delta_color="off")
        c2.metric("Best accuracy", table["Accuracy"].idxmax(), f"{table['Accuracy'].max():.4f}",
                  delta_color="off")
        c3.metric("Best AUC (OvR)", table["AUC"].idxmax(), f"{table['AUC'].max():.4f}",
                  delta_color="off")

        chart_metric = st.selectbox("Metric to plot", METRIC_ORDER, index=4, key="chart_metric")
        ordered = table[chart_metric].sort_values()
        fig, ax = plt.subplots(figsize=(8.4, 3.3))
        colors = [ACCENT if idx == champion else "#9fb8ac" for idx in ordered.index]
        ax.barh(ordered.index, ordered.values, color=colors, edgecolor="#4c5f57")
        for idx, val in enumerate(ordered.values):
            ax.text(val + 0.006, idx, f"{val:.3f}", va="center", fontsize=9)
        ax.set_xlim(0, min(1.05, ordered.max() + 0.12))
        ax.set_xlabel(chart_metric)
        ax.set_title(f"{chart_metric} by model", pad=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        st.pyplot(fig)

        st.download_button(
            "⬇️ Download this comparison table (CSV)",
            table.round(6).to_csv().encode(),
            file_name="model_comparison_current_testdata.csv",
            mime="text/csv",
        )

        with st.expander("Reference scores from the offline training run (model/metrics.json)"):
            st.dataframe(style_comparison(reference), width="stretch")
            st.caption(
                "Identical numbers confirm the deployed artefacts are exactly the ones "
                "evaluated during training on the 885-row hold-out split."
            )

# ---------------------- TAB 3 : row level --------------------------
with tab_rows:
    st.subheader(f"Per-student predictions — {model_name}")

    pred = model.predict(X)
    out = pd.DataFrame({"predicted_class": pred})
    if hasattr(model, "predict_proba"):
        proba = pd.DataFrame(model.predict_proba(X), columns=[f"P({c})" for c in model.classes_])
        out = pd.concat([out, proba.round(4)], axis=1)
        out["confidence"] = proba.max(axis=1).round(4)
    if y is not None:
        out.insert(0, "actual_class", y)
        out["correct"] = out["actual_class"] == out["predicted_class"]

    key_cols = ["Age at enrollment", "Admission grade",
                "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)",
                "Curricular units 2nd sem (grade)", "Tuition fees up to date", "Scholarship holder"]
    display = pd.concat([out, X[[c for c in key_cols if c in X.columns]]], axis=1)

    filt1, filt2 = st.columns([2, 1])
    with filt1:
        chosen = st.multiselect("Filter by predicted class", CLASSES, default=CLASSES,
                                key="class_filter")
    with filt2:
        only_wrong = st.checkbox("Show only mis-classified rows", value=False,
                                 key="only_wrong", disabled=y is None)

    view = display[display["predicted_class"].isin(chosen)]
    if only_wrong and y is not None:
        view = view[~view["correct"]]

    st.dataframe(view.head(row_limit), width="stretch")
    st.caption(f"{len(view)} row(s) match the filter; showing first {min(row_limit, len(view))}.")

    dist = pd.Series(pred).value_counts().reindex(CLASSES).fillna(0).astype(int)
    d1, d2 = st.columns([1, 2])
    with d1:
        st.markdown("**Predicted class distribution**")
        st.dataframe(dist.rename("count").to_frame(), width="stretch")
    with d2:
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        ax.bar(dist.index, dist.values, color=[PALETTE[c] for c in dist.index], edgecolor="#3d4a44")
        for i, v in enumerate(dist.values):
            ax.text(i, v + max(dist.values) * .02, str(v), ha="center", fontsize=9)
        ax.set_ylabel("students")
        sns.despine(ax=ax)
        fig.tight_layout()
        st.pyplot(fig)

    st.download_button(
        "⬇️ Download predictions (CSV)",
        display.to_csv(index=False).encode(),
        file_name=f"predictions_{model_name.split(' ')[0].lower()}.csv",
        mime="text/csv",
    )

# ---------------------- TAB 4 : about ------------------------------
with tab_about:
    st.subheader("Problem statement")
    st.write(
        "Higher-education institutions lose a substantial share of students before graduation. "
        "Using only information available at enrolment (demographics, parental background, prior "
        "qualifications, financial status) together with first- and second-semester academic "
        "records, this project predicts each student's academic outcome — **Dropout**, still "
        "**Enrolled**, or **Graduate** — so that support can be targeted early. It is a "
        "three-class, single-label classification problem."
    )

    st.subheader("Dataset description")
    info = meta["dataset"]
    a, b, c, d = st.columns(4)
    a.metric("Instances", info["instances"])
    b.metric("Features", info["features"])
    c.metric("Classes", len(info["classes"]))
    d.metric("Missing values", 0)
    st.markdown(
        f"- **Source:** {info['source']} — [dataset page]({info['url']})\n"
        f"- **Target column:** `{TARGET}` with classes {', '.join(info['classes'])}\n"
        f"- **Split:** stratified {int((1 - meta['split']['test_fraction']) * 100)}/"
        f"{int(meta['split']['test_fraction'] * 100)} → "
        f"{meta['split']['train_rows']} train / {meta['split']['test_rows']} test rows, "
        f"`random_state={meta['split']['random_state']}`\n"
        f"- **Requirement check:** {info['features']} features ≥ 12 and "
        f"{info['instances']} instances ≥ 500 ✅"
    )

    counts = pd.Series(info["class_counts"]).reindex(info["classes"])
    e1, e2 = st.columns([1, 2])
    with e1:
        st.dataframe(
            counts.rename("instances").to_frame().assign(
                share=(counts / counts.sum()).map("{:.1%}".format)
            ),
            width="stretch",
        )
    with e2:
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in counts.index],
               edgecolor="#3d4a44")
        ax.set_ylabel("instances")
        ax.set_title("Class distribution of the full dataset", pad=8)
        sns.despine(ax=ax)
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Modelling pipeline")
    prep = meta["preprocessing"]
    st.markdown(
        f"1. **Column typing** — {len(prep['one_hot'])} nominal code columns one-hot encoded "
        f"(`handle_unknown='ignore'`), {len(prep['standard_scaled'])} continuous/ordinal columns "
        f"standard-scaled, {len(prep['passthrough_binary'])} binary flags passed through.\n"
        "2. **Leak-free packaging** — every model is a single `Pipeline("
        "ColumnTransformer → estimator)`, so the scaler and encoder are fitted on training "
        "folds only and travel with the model.\n"
        "3. **Tuning** — `GridSearchCV`, 5-fold `StratifiedKFold`, scoring `f1_macro`.\n"
        "4. **Class imbalance** — `class_weight='balanced'` for Logistic Regression / Decision "
        "Tree and `'balanced_subsample'` for the Random Forest; macro averaging everywhere.\n"
        "5. **Evaluation** — untouched 20% hold-out; the app re-computes the metrics live so the "
        "numbers on screen are reproducible, not hard-coded."
    )
    st.markdown("**Tuned hyper-parameters per model**")
    st.json(meta["best_params"], expanded=False)

    st.subheader("Repository layout")
    st.code(
        "ml-assignment2/\n"
        "├── app.py                  Streamlit application (this page)\n"
        "├── ml_core.py              shared preprocessing / model / metric definitions\n"
        "├── requirements.txt        pinned dependencies\n"
        "├── README.md               problem statement, dataset, tables, observations\n"
        "├── test_data.csv           885-row stratified hold-out split\n"
        "├── train_data.csv          3539-row training split (retrain fallback)\n"
        "└── model/\n"
        "    ├── train_models.py     training + evaluation script\n"
        "    ├── train_models.ipynb  notebook version (BITS Virtual Lab run)\n"
        "    ├── metrics.json        all metrics, confusion matrices, best params\n"
        "    ├── comparison_table.csv\n"
        "    └── *.joblib            six fitted pipelines\n",
        language="text",
    )
    st.caption(
        "Author: Rishi Gudimetla · M.Tech (AIML/DSE), BITS Pilani WILP · "
        "Machine Learning Assignment 2 · executed on BITS Virtual Lab."
    )
