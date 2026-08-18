"""
=====================================================================
 ml_core.py  -  shared ML plumbing for ML Assignment 2
---------------------------------------------------------------------
 Imported by BOTH  model/train_models.py  (offline training) and
 app.py  (Streamlit front-end), so the column handling, the
 pre-processing recipe and the metric definitions can never drift
 apart between training time and serving time.
=====================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------
# Configuration shared by training and serving
# --------------------------------------------------------------------
SEED = 42
TEST_FRACTION = 0.20
TARGET = "Target"

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
RAW_CSV = PROJECT_ROOT / "rawdata" / "data.csv"
TRAIN_CSV = PROJECT_ROOT / "train_data.csv"
TEST_CSV = PROJECT_ROOT / "test_data.csv"
METRICS_JSON = MODEL_DIR / "metrics.json"

CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]

# Nominal, code-encoded attributes -> one-hot encoded
CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Already 0/1 -> passed through untouched
BINARY_COLUMNS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Human-readable notes used by the Streamlit UI
MODEL_NOTES = {
    "Logistic Regression": "Linear, class-weighted baseline; L2 regularised (C tuned by CV).",
    "Decision Tree": "Single axis-aligned tree, depth-limited to curb over-fitting.",
    "kNN": "Distance-based lazy learner on the scaled + one-hot feature space.",
    "Naive Bayes": "GaussianNB; assumes conditional independence of features.",
    "Random Forest (Ensemble)": "500 bagged trees with balanced subsampling.",
    "Gradient Boosting (Ensemble, extra)": "Additive boosted stumps/shallow trees (optional 6th model).",
}


# --------------------------------------------------------------------
# Column hygiene
# --------------------------------------------------------------------
def tidy_column(name: str) -> str:
    """UCI headers carry stray quotes / tabs / BOM, e.g. '"Daytime/evening attendance\\t"'."""
    return str(name).replace('"', "").replace("\t", "").replace("\ufeff", "").strip()


def tidy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [tidy_column(c) for c in out.columns]
    return out


def load_raw_dataset() -> pd.DataFrame:
    """Read the semicolon-separated UCI source file."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw UCI file not found at {RAW_CSV}. Download it with:\n"
            '  curl -L -o uci697.zip "https://archive.ics.uci.edu/static/public/697/'
            'predict+students+dropout+and+academic+success.zip"\n'
            "  unzip uci697.zip -d rawdata"
        )
    return tidy_frame(pd.read_csv(RAW_CSV, sep=";", encoding="utf-8-sig"))


# --------------------------------------------------------------------
# Pre-processing
# --------------------------------------------------------------------
def split_column_groups(feature_frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    categorical = [c for c in CATEGORICAL_COLUMNS if c in feature_frame.columns]
    binary = [c for c in BINARY_COLUMNS if c in feature_frame.columns]
    numeric = [c for c in feature_frame.columns if c not in categorical + binary]
    return categorical, numeric, binary


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    categorical, numeric, binary = split_column_groups(feature_frame)
    return ColumnTransformer(
        transformers=[
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", StandardScaler(), numeric),
            ("binary", "passthrough", binary),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------
# Model zoo:  5 required models + 1 optional extra ensemble
# --------------------------------------------------------------------
def model_zoo() -> dict[str, dict]:
    return {
        "Logistic Regression": {
            "file": "logistic_regression.joblib",
            "estimator": LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=SEED
            ),
            "grid": {"clf__C": [0.05, 0.5, 1.0, 5.0]},
        },
        "Decision Tree": {
            "file": "decision_tree.joblib",
            "estimator": DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            "grid": {
                "clf__max_depth": [4, 6, 8, 12, None],
                "clf__min_samples_leaf": [1, 5, 20],
                "clf__criterion": ["gini", "entropy"],
            },
        },
        "kNN": {
            "file": "knn.joblib",
            "estimator": KNeighborsClassifier(),
            "grid": {
                "clf__n_neighbors": [5, 11, 17, 25, 35],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        },
        "Naive Bayes": {
            "file": "naive_bayes.joblib",
            "estimator": GaussianNB(),
            "grid": {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
        },
        "Random Forest (Ensemble)": {
            "file": "random_forest.joblib",
            "estimator": RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            ),
            "grid": {
                "clf__max_depth": [None, 12, 20],
                "clf__min_samples_leaf": [1, 3],
                "clf__max_features": ["sqrt", 0.3],
            },
        },
        # The assignment text mentions "6 ML models" while listing 5, so a
        # second ensemble is added for completeness and clearly marked "extra".
        "Gradient Boosting (Ensemble, extra)": {
            "file": "gradient_boosting.joblib",
            "estimator": GradientBoostingClassifier(random_state=SEED),
            "grid": {
                "clf__n_estimators": [200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
        },
    }


def build_pipeline(model_name: str, feature_frame: pd.DataFrame, params: dict | None = None) -> Pipeline:
    """Rebuild an untrained pipeline; used by training and by the app's retrain fallback."""
    spec = model_zoo()[model_name]
    estimator = spec["estimator"]
    if params:
        estimator = estimator.set_params(**params)
    return Pipeline([("prep", build_preprocessor(feature_frame)), ("clf", estimator)])


# --------------------------------------------------------------------
# Metrics  (works for binary and multi-class targets)
# --------------------------------------------------------------------
def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, class_labels: list[str]) -> dict:
    """Return the six assignment metrics plus confusion matrix / per-class report."""
    y_pred = model.predict(X_test)
    n_classes = len(class_labels)
    average = "binary" if n_classes == 2 else "macro"

    kwargs: dict = {"average": average, "zero_division": 0}
    if n_classes == 2:
        kwargs["pos_label"] = class_labels[-1]

    auc = float("nan")
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            if n_classes == 2:
                auc = roc_auc_score((y_test == class_labels[-1]).astype(int), proba[:, 1])
            else:
                auc = roc_auc_score(
                    y_test, proba, multi_class="ovr", average="macro", labels=class_labels
                )
        except ValueError:
            auc = float("nan")  # e.g. a class missing from the uploaded slice

    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "AUC": float(auc),
        "Precision": float(precision_score(y_test, y_pred, **kwargs)),
        "Recall": float(recall_score(y_test, y_pred, **kwargs)),
        "F1": float(f1_score(y_test, y_pred, **kwargs)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=class_labels).tolist(),
        "per_class": classification_report(
            y_test, y_pred, labels=class_labels, zero_division=0, output_dict=True
        ),
        "report_text": classification_report(
            y_test, y_pred, labels=class_labels, zero_division=0, digits=4
        ),
        "y_pred": y_pred,
    }


METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def metrics_to_frame(results: dict[str, dict]) -> pd.DataFrame:
    """{model: metric dict} -> tidy comparison DataFrame."""
    rows = {
        name: {m: vals.get(m, np.nan) for m in METRIC_ORDER}
        for name, vals in results.items()
    }
    return pd.DataFrame(rows).T[METRIC_ORDER]


# --------------------------------------------------------------------
# Incoming-CSV validation (pure function so it is unit-testable)
# --------------------------------------------------------------------
def prepare_test_frame(
    raw: pd.DataFrame, expected: list[str]
) -> tuple[pd.DataFrame | None, pd.Series | None, list[str], list[str]]:
    """Validate and clean a user-supplied test CSV.

    Returns (X, y, notes, errors).
      * y is None when the CSV carries no ``Target`` column (prediction-only mode).
      * X is None when ``errors`` is non-empty, i.e. the file cannot be scored.
    """
    notes: list[str] = []
    errors: list[str] = []
    frame = tidy_frame(raw)

    y = None
    if TARGET in frame.columns:
        y = frame[TARGET].astype("string").str.strip()
        frame = frame.drop(columns=[TARGET])
    else:
        notes.append(
            f"No `{TARGET}` column found - running in prediction-only mode "
            "(evaluation metrics need ground-truth labels)."
        )

    missing = [c for c in expected if c not in frame.columns]
    if missing:
        errors.append(
            f"The CSV is missing {len(missing)} required feature column(s): "
            + ", ".join(missing[:8])
            + ("..." if len(missing) > 8 else "")
        )
        return None, None, notes, errors

    extra = [c for c in frame.columns if c not in expected]
    if extra:
        notes.append(f"Ignored {len(extra)} unexpected column(s): {', '.join(extra[:6])}")

    X = frame[expected].apply(pd.to_numeric, errors="coerce")

    bad = X.isna().any(axis=1)
    if y is not None:
        bad = bad | y.isna()
    if bool(bad.any()):
        notes.append(f"Dropped {int(bad.sum())} row(s) with missing / non-numeric values.")
        X = X.loc[~bad]
        if y is not None:
            y = y.loc[~bad]

    if X.empty:
        errors.append("No usable rows left after cleaning - please check the file.")
        return None, None, notes, errors

    if y is not None:
        unknown = sorted(set(y.dropna()) - set(CLASS_ORDER))
        if unknown:
            notes.append(
                "Target values not seen during training will count as errors: "
                + ", ".join(map(str, unknown[:5]))
            )

    y_out = None if y is None else pd.Series(y.to_numpy(), name=TARGET).astype(object)
    return X.reset_index(drop=True), y_out, notes, errors
