"""
Verification suite for ML Assignment 2.

Run from the project root:
    python -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_core import (  # noqa: E402
    CLASS_ORDER,
    METRIC_ORDER,
    METRICS_JSON,
    MODEL_DIR,
    TARGET,
    TEST_CSV,
    TRAIN_CSV,
    evaluate,
    prepare_test_frame,
    tidy_frame,
)

META = json.loads(METRICS_JSON.read_text())
FEATURES = META["feature_columns"]


# ------------------------------------------------------------------ data
def test_dataset_meets_assignment_minimums():
    assert META["dataset"]["features"] >= 12, "assignment requires >= 12 features"
    assert META["dataset"]["instances"] >= 500, "assignment requires >= 500 instances"


def test_split_files_exist_and_are_disjoint():
    train = tidy_frame(pd.read_csv(TRAIN_CSV))
    test = tidy_frame(pd.read_csv(TEST_CSV))
    assert len(train) == META["split"]["train_rows"]
    assert len(test) == META["split"]["test_rows"]
    assert TARGET in train.columns and TARGET in test.columns
    # no row of the hold-out set leaked into training
    overlap = pd.merge(train, test, how="inner")
    assert overlap.empty, f"{len(overlap)} test rows also appear in train_data.csv"


def test_test_set_is_stratified():
    test = pd.read_csv(TEST_CSV)
    full_share = pd.Series(META["dataset"]["class_counts"])
    full_share = full_share / full_share.sum()
    test_share = test[TARGET].value_counts(normalize=True)
    for cls in CLASS_ORDER:
        assert abs(test_share[cls] - full_share[cls]) < 0.02


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("name,filename", list(META["models"].items()))
def test_every_model_file_loads_and_predicts(name, filename):
    path = MODEL_DIR / filename
    assert path.exists(), f"missing artefact for {name}"
    model = joblib.load(path)
    test = tidy_frame(pd.read_csv(TEST_CSV))
    X, y = test.drop(columns=[TARGET]), test[TARGET]
    preds = model.predict(X.head(20))
    assert len(preds) == 20
    assert set(preds).issubset(set(CLASS_ORDER))


def test_five_required_models_are_present():
    required = {"Logistic Regression", "Decision Tree", "kNN",
                "Naive Bayes", "Random Forest (Ensemble)"}
    assert required.issubset(set(META["models"])), "a mandated model is missing"


@pytest.mark.parametrize("name,filename", list(META["models"].items()))
def test_saved_metrics_are_reproducible(name, filename):
    """Recomputing the six metrics from the artefact must match metrics.json."""
    model = joblib.load(MODEL_DIR / filename)
    test = tidy_frame(pd.read_csv(TEST_CSV))
    X, y = test.drop(columns=[TARGET]), test[TARGET]
    fresh = evaluate(model, X, y, CLASS_ORDER)
    for metric in METRIC_ORDER:
        expected = META["metrics"][name][metric]
        assert expected is not None
        assert fresh[metric] == pytest.approx(expected, abs=1e-6), f"{name} / {metric}"


def test_all_six_metrics_present_for_every_model():
    for name, block in META["metrics"].items():
        for metric in METRIC_ORDER:
            assert block.get(metric) is not None, f"{name} is missing {metric}"
            assert 0.0 <= block[metric] <= 1.0 or metric == "MCC"


# ------------------------------------------------------- CSV validation
def _sample(n: int = 30) -> pd.DataFrame:
    return pd.read_csv(TEST_CSV).head(n)


def test_prepare_accepts_the_bundled_test_file():
    X, y, notes, errors = prepare_test_frame(_sample(), FEATURES)
    assert not errors
    assert y is not None and len(X) == 30
    assert list(X.columns) == FEATURES


def test_prepare_handles_missing_target_as_prediction_only():
    X, y, notes, errors = prepare_test_frame(_sample().drop(columns=[TARGET]), FEATURES)
    assert not errors and y is None
    assert any("prediction-only" in n for n in notes)


def test_prepare_rejects_missing_feature_columns():
    X, y, notes, errors = prepare_test_frame(_sample().drop(columns=["Admission grade"]), FEATURES)
    assert X is None and errors and "Admission grade" in errors[0]


def test_prepare_ignores_extra_columns_and_untidy_headers():
    frame = _sample()
    frame["some_extra_column"] = 1
    frame = frame.rename(columns={"Admission grade": '"Admission grade\t"'})
    X, y, notes, errors = prepare_test_frame(frame, FEATURES)
    assert not errors
    assert "Admission grade" in X.columns
    assert any("Ignored 1 unexpected" in n for n in notes)


def test_prepare_drops_rows_with_missing_values():
    frame = _sample()
    frame["Admission grade"] = frame["Admission grade"].astype(float)
    frame["Age at enrollment"] = frame["Age at enrollment"].astype(object)
    frame.loc[0, "Admission grade"] = np.nan
    frame.loc[1, "Age at enrollment"] = "not-a-number"
    X, y, notes, errors = prepare_test_frame(frame, FEATURES)
    assert not errors and len(X) == 28
    assert any("Dropped 2 row" in n for n in notes)


def test_models_score_an_uploaded_subset_without_crashing():
    """A small single-class slice must not raise; AUC simply becomes NaN."""
    frame = pd.read_csv(TEST_CSV)
    one_class = frame[frame[TARGET] == "Graduate"].head(15)
    X, y, notes, errors = prepare_test_frame(one_class, FEATURES)
    assert not errors
    model = joblib.load(MODEL_DIR / META["models"]["Logistic Regression"])
    scored = evaluate(model, X, y, CLASS_ORDER)
    assert np.isnan(scored["AUC"])
    assert 0.0 <= scored["Accuracy"] <= 1.0


# ------------------------------------------------------- Streamlit app
def _app(timeout: int = 300):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=timeout).run()


def test_streamlit_app_runs_without_exceptions():
    at = _app()
    assert not at.exception, at.exception
    assert len(at.tabs) >= 4, "expected the four content tabs to render"


def test_streamlit_app_switches_between_all_models():
    at = _app()
    assert not at.exception, at.exception
    for name in META["models"]:
        at.selectbox(key="model_choice").set_value(name).run()
        assert not at.exception, f"{name}: {at.exception}"
        assert at.selectbox(key="model_choice").value == name


def test_streamlit_app_toggles_display_options():
    at = _app()
    at.checkbox(key="norm_cm").set_value(True).run()          # normalise confusion matrix
    assert not at.exception, at.exception
    at.checkbox(key="show_roc").set_value(False).run()        # hide ROC curves
    assert not at.exception, at.exception
    at.selectbox(key="chart_metric").set_value("MCC").run()   # switch comparison chart metric
    assert not at.exception, at.exception
    at.slider(key="row_limit").set_value(50).run()
    assert not at.exception, at.exception


def test_streamlit_app_exposes_required_features():
    """a) upload widget  b) model dropdown  c) metrics  d) confusion matrix / report."""
    at = _app()

    # b) model selection dropdown listing every trained model
    dropdown = at.selectbox(key="model_choice")
    assert list(dropdown.options) == list(META["models"].keys())

    # c) all six evaluation metrics displayed as cards
    labels = {m.label for m in at.metric}
    assert set(METRIC_ORDER).issubset(labels), f"metric cards missing: {labels}"

    # d) confusion matrix figure + classification report rendered
    assert len(at.get("image")) >= 2, "confusion matrix / ROC figures missing"
    assert any("Classification report" in md.value for md in at.markdown), "report missing"

    # a) CSV upload widget appears when the upload source is selected
    at.radio(key="data_source").set_value("Upload my own test CSV").run()
    assert not at.exception, at.exception
    uploaders = at.get("file_uploader")
    assert uploaders and any("Upload" in u.label for u in uploaders), "CSV upload widget missing"


def test_streamlit_comparison_tab_scores_every_model():
    """The comparison tab must show a row per model on the live test data."""
    at = _app()
    assert not at.exception, at.exception
    tables = at.dataframe
    assert tables, "no dataframes rendered"
    indexes = [list(t.value.index) for t in tables if hasattr(t.value, "index")]
    comparison = [ix for ix in indexes if set(META["models"]).issubset(set(map(str, ix)))]
    assert comparison, f"no table contains all six models; found {indexes}"


def _upload(at, frame: pd.DataFrame, name: str):
    at.radio(key="data_source").set_value("Upload my own test CSV").run()
    at.get("file_uploader")[0].upload(name, frame.to_csv(index=False).encode(), "text/csv").run()
    return at


def test_streamlit_app_scores_an_uploaded_labelled_csv():
    at = _app()
    subset = pd.read_csv(TEST_CSV).head(60)
    _upload(at, subset, "mini_test.csv")
    assert not at.exception, at.exception
    assert any("60" in c.value and "uploaded file" in c.value for c in at.caption)
    labels = {m.label for m in at.metric}
    assert set(METRIC_ORDER).issubset(labels)


def test_streamlit_app_handles_unlabelled_upload():
    at = _app()
    subset = pd.read_csv(TEST_CSV).head(40).drop(columns=[TARGET])
    _upload(at, subset, "nolabel.csv")
    assert not at.exception, at.exception
    assert any("prediction-only" in i.value for i in at.info)


def test_streamlit_app_rejects_a_malformed_upload_gracefully():
    at = _app()
    subset = pd.read_csv(TEST_CSV).head(10).drop(columns=["Admission grade", "GDP"])
    _upload(at, subset, "broken.csv")
    assert not at.exception, at.exception          # handled, not crashed
    assert any("missing" in e.value for e in at.error), [e.value for e in at.error]
