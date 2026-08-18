"""
=====================================================================
 ML Assignment 2  -  Student Academic Outcome Classifier
 M.Tech (AIML / DSE) - Work Integrated Learning Programmes Division
---------------------------------------------------------------------
 Trains + evaluates the 5 required classification models (plus one
 optional extra ensemble) on the UCI dataset
 "Predict Students' Dropout and Academic Success" (ID 697) and writes:

   train_data.csv                 stratified 80% training split
   test_data.csv                  stratified 20% hold-out split (used by app.py)
   model/<model>.joblib           fitted end-to-end pipelines
   model/metrics.json             every metric, CM, per-class report, best params
   model/comparison_table.csv     the README comparison table

 Run from the project root:   python model/train_models.py
=====================================================================
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import ml_core from root
from ml_core import (  # noqa: E402
    MODEL_DIR,
    SEED,
    TARGET,
    TEST_CSV,
    TEST_FRACTION,
    TRAIN_CSV,
    build_pipeline,
    evaluate,
    load_raw_dataset,
    metrics_to_frame,
    model_zoo,
    split_column_groups,
)

warnings.filterwarnings("ignore")

COMPRESSION = 3  # keeps the Random Forest artefact GitHub-friendly


def describe_dataset(frame: pd.DataFrame) -> None:
    print("=" * 72)
    print("DATASET SUMMARY  -  UCI 697: Predict Students' Dropout and Academic Success")
    print("=" * 72)
    print(f"  instances ............ {frame.shape[0]}")
    print(f"  attributes ........... {frame.shape[1] - 1} features + 1 target")
    print(f"  missing values ....... {int(frame.isna().sum().sum())}")
    print(f"  duplicated rows ...... {int(frame.duplicated().sum())}")
    counts = frame[TARGET].value_counts()
    print("  class distribution ...")
    for label, n in counts.items():
        print(f"      {label:<10} {n:>5}  ({n / len(frame):6.2%})")
    print(f"  imbalance ratio ...... {counts.max() / counts.min():.2f} : 1")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_raw_dataset()
    describe_dataset(data)

    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED
    )
    print(f"\nStratified hold-out -> train {X_train.shape[0]} rows | test {X_test.shape[0]} rows")

    X_train.assign(**{TARGET: y_train}).to_csv(TRAIN_CSV, index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(TEST_CSV, index=False)
    print(f"  wrote {TRAIN_CSV.name} and {TEST_CSV.name}")

    categorical, numeric, binary = split_column_groups(X_train)
    print("\nPre-processing plan")
    print(f"  one-hot encoded  : {len(categorical)} nominal columns")
    print(f"  standard scaled  : {len(numeric)} continuous / ordinal columns")
    print(f"  passed through   : {len(binary)} binary flags")

    class_labels = sorted(y.unique())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    zoo = model_zoo()
    results, tuned, per_class, confusion, cv_scores = {}, {}, {}, {}, {}

    print("\n" + "=" * 72)
    print("TRAINING  (5-fold stratified GridSearchCV, scoring = macro F1)")
    print("=" * 72)

    for name, spec in zoo.items():
        started = time.perf_counter()
        search = GridSearchCV(
            build_pipeline(name, X_train),
            spec["grid"],
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        scored = evaluate(best, X_test, y_test, class_labels)
        per_class[name] = scored.pop("per_class")
        confusion[name] = scored.pop("confusion_matrix")
        scored.pop("report_text")
        scored.pop("y_pred")
        results[name] = scored
        tuned[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        cv_scores[name] = float(search.best_score_)

        joblib.dump(best, MODEL_DIR / spec["file"], compress=COMPRESSION)
        size_mb = (MODEL_DIR / spec["file"]).stat().st_size / 1e6
        print(
            f"\n  {name}"
            f"\n    best params      : {tuned[name]}"
            f"\n    cv macro-F1      : {search.best_score_:.4f}"
            f"\n    test accuracy    : {scored['Accuracy']:.4f}"
            f"\n    test AUC (ovr)   : {scored['AUC']:.4f}"
            f"\n    test macro-F1    : {scored['F1']:.4f}"
            f"\n    test MCC         : {scored['MCC']:.4f}"
            f"\n    saved            : model/{spec['file']} "
            f"({size_mb:.2f} MB, {time.perf_counter() - started:.1f}s)"
        )

    table = metrics_to_frame(results)
    print("\n" + "=" * 72)
    print("COMPARISON TABLE  (885-row hold-out test set, macro averaging, AUC = OvR)")
    print("=" * 72)
    print(table.round(4).to_string())

    winner = str(table["F1"].idxmax())
    print(f"\nOverall winner (macro-F1): {winner}")

    # ---- design facts quoted in the README: keep them generated, never hand-typed
    reference_model = joblib.load(MODEL_DIR / zoo["Logistic Regression"]["file"])
    expanded_dim = int(reference_model.named_steps["prep"].transform(X_test.head(50)).shape[1])
    inference_ms = {}
    for name, spec in zoo.items():
        mdl = joblib.load(MODEL_DIR / spec["file"])
        mdl.predict(X_test.head(5))                      # warm up
        clock = time.perf_counter()
        for _ in range(3):
            mdl.predict(X_test)
        inference_ms[name] = round((time.perf_counter() - clock) / 3 * 1000, 1)
    print(f"\nExpanded feature dimensionality after encoding: {expanded_dim}")
    print("Inference time for the full test set (ms): "
          + ", ".join(f"{k.split(' (')[0]} {v}" for k, v in inference_ms.items()))

    metadata = {
        "dataset": {
            "name": "Predict Students' Dropout and Academic Success",
            "source": "UCI Machine Learning Repository, dataset ID 697",
            "url": "https://archive.ics.uci.edu/dataset/697/"
                   "predict+students+dropout+and+academic+success",
            "instances": int(data.shape[0]),
            "features": int(data.shape[1] - 1),
            "classes": [str(c) for c in class_labels],
            "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        },
        "split": {
            "test_fraction": TEST_FRACTION,
            "random_state": SEED,
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "stratified": True,
        },
        "averaging": "macro (3-class problem); AUC = one-vs-rest macro",
        "preprocessing": {
            "one_hot": categorical,
            "standard_scaled": numeric,
            "passthrough_binary": binary,
            "expanded_dimensionality": expanded_dim,
        },
        "artefact_size_mb": {
            name: round((MODEL_DIR / spec["file"]).stat().st_size / 1e6, 2)
            for name, spec in zoo.items()
        },
        "inference_ms_full_testset": inference_ms,
        "feature_columns": list(X.columns),
        "target_column": TARGET,
        "models": {name: zoo[name]["file"] for name in results},
        "best_params": tuned,
        "cv_macro_f1": cv_scores,
        "metrics": {
            name: {m: (None if np.isnan(v) else round(float(v), 6)) for m, v in vals.items()}
            for name, vals in results.items()
        },
        "confusion_matrices": confusion,
        "per_class_report": per_class,
        "winner": winner,
        "sklearn_version": __import__("sklearn").__version__,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2))
    table.round(4).to_csv(MODEL_DIR / "comparison_table.csv")
    print("\nWrote model/metrics.json and model/comparison_table.csv")
    print("Done.")


if __name__ == "__main__":
    main()
