# Student Outcome Radar — Machine Learning Assignment 2

**M.Tech (AIML / DSE) · Work Integrated Learning Programmes Division, BITS Pilani**
Predicting whether a higher-education student will **drop out**, stay **enrolled**, or
**graduate**, with six classification models and an interactive Streamlit front-end.

| | |
|---|---|
| **GitHub repository** | https://github.com/rishigudimetla/2025AC05991 |
| **Live Streamlit app** | https://rishigudimetla-2025ac05991-app-26xzre.streamlit.app |
| **Dataset** | [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) — UCI Machine Learning Repository, dataset ID 697 |
| **Best model** | **Random Forest (Ensemble)** (accuracy 0.7684, macro-F1 0.7171, MCC 0.6240) |
| **Executed on** | BITS Virtual Lab (screenshot included in the submitted PDF) |

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
49.9% accuracy by a clear margin; (2) maximise **macro-F1** and
**MCC** so the minority *Enrolled* class cannot be sacrificed; (3) keep *Dropout* recall high,
because a missed at-risk student is costlier than an unnecessary counselling session.

---

## b. Dataset description

**Predict Students' Dropout and Academic Success** — UCI Machine Learning Repository, dataset ID 697
Source URL: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

| Property | Value | Assignment requirement | Status |
|---|---|---|---|
| Instances | **4,424** | >= 500 | Met |
| Features | **36** | >= 12 | Met |
| Classes | 3 (Dropout, Enrolled, Graduate) | binary *or* multi-class | Met (multi-class) |
| Missing values | 0 | - | Met, no imputation needed |
| Duplicate rows | 0 | - | Met |
| Source repository | UCI Machine Learning Repository (ID 697) | Kaggle or UCI | Met |
| Licence | CC BY 4.0 | - | Met, redistributable |

### Target variable

| Class | Instances | Share | Meaning |
|---|---|---|---|
| Dropout | 1421 | 32.1% | left the programme before completing it |
| Enrolled | 794 | 17.9% | still studying at the end of the normal course duration |
| Graduate | 2209 | 49.9% | completed the programme on time |

The classes are imbalanced at **2.78 : 1**
(largest : smallest). A model that always answered *Graduate* would already score
49.9% accuracy — the reason macro-F1 and MCC, not accuracy, are
used to pick the winner.

### Feature groups (36 predictors)

| Group | Count | Examples | Handling |
|---|---|---|---|
| Nominal codes | 9 | `Course`, `Application mode`, `Mother's / Father's qualification`, `Mother's / Father's occupation`, `Nacionality`, `Previous qualification`, `Marital status` | **One-hot encoded** (`handle_unknown='ignore'`) — the integers are labels, not magnitudes, so scaling them would invent a false ordering |
| Continuous / ordinal | 19 | `Admission grade`, `Age at enrollment`, `Curricular units 1st/2nd sem (enrolled · evaluations · approved · grade · credited · without evaluations)`, `Unemployment rate`, `Inflation rate`, `GDP` | **StandardScaler** — required by Logistic Regression and kNN, harmless for the tree models |
| Binary flags | 8 | `Debtor`, `Tuition fees up to date`, `Scholarship holder`, `Displaced`, `Gender`, `Daytime/evening attendance`, `Educational special needs`, `International` | **Passed through** — already 0/1 |

### Train / test split

| | Rows | Purpose |
|---|---|---|
| `train_data.csv` | 3,539 (80%) | model fitting + 5-fold CV hyper-parameter search |
| `test_data.csv` | 885 (20%) | untouched hold-out; this is the file the Streamlit app scores |

Split is **stratified** on the target with `random_state=42`, so the class
proportions above are preserved in both files and every run is reproducible. Only
`test_data.csv` is uploaded to / shipped with the app, in line with the instruction to keep
the free Streamlit tier light.

---

## c. GitHub repository link

**https://github.com/rishigudimetla/2025AC05991**

```
2025AC05991/
├── app.py                    # Streamlit application (deployed entry point)
├── ml_core.py                # shared preprocessing, model zoo, metric definitions
├── requirements.txt          # pinned dependencies
├── README.md                 # this file
├── test_data.csv             # 885-row stratified hold-out split (upload me in the app)
├── train_data.csv            # 3539-row training split (retrain fallback)
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
├── tests/test_project.py     # 37 automated checks (data, models, metrics, UI)
└── tools/                    # README + submission-PDF generators
```

### Reproduce from scratch

```bash
git clone https://github.com/rishigudimetla/2025AC05991
cd 2025AC05991
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
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

All six pipelines are trained on the **same** 3539-row training split and scored on
the **same** untouched 885-row hold-out split. Each model is packaged as a single
`Pipeline(ColumnTransformer → estimator)`, so the encoder and scaler are fitted inside every
CV fold — there is **no leakage** from the test set into preprocessing — and the artefact that
is deployed is byte-for-byte the artefact that was evaluated.

| # | Model | scikit-learn estimator | Selected hyper-parameters (5-fold CV, `f1_macro`) | CV macro-F1 |
|---|---|---|---|---|
| 1 | Logistic Regression | `LogisticRegression(class_weight='balanced', max_iter=3000)` | `C=0.05` | 0.7125 |
| 2 | Decision Tree | `DecisionTreeClassifier(class_weight='balanced')` | `criterion=gini`, `max_depth=6`, `min_samples_leaf=1` | 0.6700 |
| 3 | kNN | `KNeighborsClassifier()` | `n_neighbors=5`, `p=1`, `weights=uniform` | 0.6222 |
| 4 | Naive Bayes | `GaussianNB()` | `var_smoothing=0.1` | 0.6323 |
| 5 | Random Forest (Ensemble) | `RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample')` | `max_depth=None`, `max_features=sqrt`, `min_samples_leaf=3` | 0.7154 |
| 6 | Gradient Boosting (Ensemble, *extra*) | `GradientBoostingClassifier()` | `learning_rate=0.1`, `max_depth=3`, `n_estimators=200` | 0.7128 |

> Models 1–5 are the five mandated by the brief. Model 6 is added because the assignment text
> refers to *"all the 6 ML models"* while listing five, so a second ensemble is supplied to
> cover both readings; it is labelled *extra* everywhere and never replaces a required model.

After encoding, all models see the same **236-dimensional**
input (9 nominal columns expanded to one-hot, plus
19 scaled and 8 binary columns).
Deployment cost differs sharply, which matters on a free hosting tier:

| Model | Artefact size | Inference time, 885 rows |
|---|---|---|
| Logistic Regression | 0.01 MB | 2.5 ms |
| Decision Tree | 0.01 MB | 2.6 ms |
| kNN | 0.39 MB | 77.2 ms |
| Naive Bayes | 0.01 MB | 3.2 ms |
| Random Forest (Ensemble) | 12.53 MB | 39.6 ms |
| Gradient Boosting (Ensemble, extra) | 0.20 MB | 6.7 ms |

### How each metric is computed

The target has three classes, so the assignment's six metrics are reported as follows:

| Metric | Definition used | Why |
|---|---|---|
| Accuracy | `accuracy_score` | overall hit rate; baseline to beat is 49.9% |
| AUC | `roc_auc_score(..., multi_class='ovr', average='macro')` | one-vs-rest AUC per class, then unweighted mean — the standard multi-class extension; uses `predict_proba`, so it judges the ranking rather than the argmax |
| Precision | `precision_score(..., average='macro')` | unweighted mean over the three classes |
| Recall | `recall_score(..., average='macro')` | unweighted mean over the three classes |
| F1 | `f1_score(..., average='macro')` | harmonic mean of the two above, per class then averaged |
| MCC | `matthews_corrcoef` | chance-corrected correlation over the full 3×3 matrix; the most honest single number under imbalance |

**Macro** (not *weighted*) averaging is used deliberately: it refuses to let the
794-instance *Enrolled* class be drowned out by the
2209-instance *Graduate* class.

### Comparison table — evaluation metrics for all models

Hold-out test set, 885 rows never seen during training or tuning.
Best value in each column is **bold**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7322 | 0.8859 | 0.7039 | 0.7062 | 0.6940 | 0.5856 |
| Decision Tree | 0.6486 | 0.8240 | 0.6660 | 0.6580 | 0.6314 | 0.4929 |
| kNN | 0.6938 | 0.7979 | 0.6294 | 0.5978 | 0.6050 | 0.4876 |
| Naive Bayes | 0.6814 | 0.8163 | 0.6200 | 0.5948 | 0.6009 | 0.4725 |
| Random Forest (Ensemble) | **0.7684** | **0.8876** | **0.7203** | **0.7164** | **0.7171** | **0.6240** |
| Gradient Boosting (Ensemble, extra) | 0.7582 | 0.8820 | 0.6977 | 0.6783 | 0.6852 | 0.6000 |

The same table restricted to the five mandated models:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7322 | 0.8859 | 0.7039 | 0.7062 | 0.6940 | 0.5856 |
| Decision Tree | 0.6486 | 0.8240 | 0.6660 | 0.6580 | 0.6314 | 0.4929 |
| kNN | 0.6938 | 0.7979 | 0.6294 | 0.5978 | 0.6050 | 0.4876 |
| Naive Bayes | 0.6814 | 0.8163 | 0.6200 | 0.5948 | 0.6009 | 0.4725 |
| Random Forest (Ensemble) | **0.7684** | **0.8876** | **0.7203** | **0.7164** | **0.7171** | **0.6240** |

*(Generated from `model/metrics.json`; the Streamlit app recomputes these numbers live from
the saved artefacts, and `tests/test_project.py` asserts that the recomputed values match this
table to 1e-6.)*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strongest non-ensemble model: accuracy 0.7322, macro-F1 0.6940, MCC 0.5856, and an AUC of 0.8859 that is essentially level with the Random Forest (0.8876). The class boundaries are therefore largely linear once the semester-approval features are scaled. `class_weight='balanced'` lifts *Enrolled* recall to 0.6415 but its precision stays low (0.4163), so it over-predicts the middle class. Cheapest model to train and the most interpretable (signed coefficients), which makes it a sensible fallback if the deployment budget is tight. |
| Decision Tree | **Lowest accuracy of the six** (0.6486, MCC 0.4929). CV chose `criterion=gini`, `max_depth=6`, `min_samples_leaf=1`, i.e. pruning was necessary - an unpruned tree over-fitted the one-hot encoded parental-qualification columns. It buys the highest *Enrolled* recall of all models (0.7107) at a precision of only 0.3404, and it drags *Graduate* recall down to 0.6471 - it happily trades away the majority class to chase the middle one. A single tree is high-variance here; it is best read as the interpretable building block that the ensembles average away. |
| kNN | Accuracy 0.6938 looks respectable but macro-F1 is only 0.6050 and AUC is the lowest of the six (0.7979): one-hot encoding expands the input to 236 dimensions, where Manhattan/Euclidean neighbourhoods stop being informative (curse of dimensionality). It collapses onto the majority class - *Graduate* recall 0.8914 versus *Enrolled* recall only 0.2893. It is also the one model with no `class_weight` option, so the 2.78:1 imbalance hits it hardest, and as a lazy learner it is the slowest at inference (77.2 ms for the 885-row test set versus 2.5 ms for Logistic Regression). |
| Naive Bayes | Accuracy 0.6814, macro-F1 0.6009 - the lowest macro-F1 and MCC of the six. The Gaussian conditional-independence assumption is clearly violated: the 1st- and 2nd-semester enrolled/approved/grade columns are strongly correlated, so evidence is double-counted and the posteriors are over-confident. Only 0.3396 of *Enrolled* students are recovered. Its AUC (0.8163) still beats kNN, so the ranking it produces is usable even though its hard decisions are not. Trains in well under a second - a useful baseline. |
| Random Forest (Ensemble) | **Best model on every one of the six metrics**: accuracy 0.7684, AUC 0.8876, macro-F1 0.7171, MCC 0.6240. Bagging 500 trees removes the single tree's variance (+0.0857 macro-F1 over the Decision Tree) while `balanced_subsample` keeps the minority class visible: it is the only model that holds *Enrolled* precision at or above 0.50 (0.5000) while still recalling 0.5472 of them. Also best on the two classes that matter operationally - *Dropout* recall 0.7289 and *Graduate* recall 0.8733. Cost: the largest artefact (~13 MB) and the slowest training run. |
| Gradient Boosting (Ensemble, extra) | Included because the brief mentions "6 ML models" while listing five. Accuracy 0.7582 and AUC 0.8820 come close to the Random Forest, and it gives the best *Graduate* recall overall (0.9027), but boosting without class weighting sacrifices the minority class (*Enrolled* recall 0.3962), so macro-F1 drops to 0.6852. Confirms that on this dataset the gain comes from *ensembling*, and that handling the imbalance is what separates the two ensembles. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** - it tops all six metrics simultaneously (accuracy 0.7684, AUC 0.8876, macro-F1 0.7171, MCC 0.6240) and its 5-fold CV score (0.7154) is consistent with its hold-out score, so the win is not a lucky split. MCC of 0.6240 versus 0.4725 for the weakest model shows the margin is substantial on a chance-corrected measure too. |

### What the errors look like

Confusion matrix of the winning model (Random Forest (Ensemble)), rows = actual, columns = predicted:

| actual ↓ / predicted → | Dropout | Enrolled | Graduate |
|---|---|---|---|
| **Dropout** | 207 | 44 | 33 |
| **Enrolled** | 33 | 87 | 39 |
| **Graduate** | 13 | 43 | 386 |

Three findings that hold across every model:

1. **Dropout vs Graduate is nearly solved.** Only
   33 true *Dropout* students are called *Graduate*
   and 13 the other way round — the two extremes are
   well separated by the semester-approval and fee-status features.
2. **All the difficulty sits in *Enrolled*.** It is the minority class *and* semantically
   in-between; every model's worst per-class F1 is *Enrolled*
   (best being 0.5225 for the
   Random Forest). Anyone using this model in practice should treat *Enrolled* as
   "needs a human look", not as a confident prediction.
3. **Handling the imbalance matters more than raw model power.** The class-weighted Logistic
   Regression (macro-F1 0.6940) beats the unweighted Gradient
   Boosting (0.6852) on macro-F1 despite being far simpler.

---

## Streamlit application

**Live app: https://rishigudimetla-2025ac05991-app-26xzre.streamlit.app** — deployed on Streamlit Community Cloud (free tier).

Features, mapped to the assignment's requirements:

| Requirement | Where it is in the app | Detail |
|---|---|---|
| **a. Dataset upload option (CSV)** | Sidebar → *1 · Test data* → **Upload my own test CSV** | Accepts any CSV with the 36 feature columns. Header quirks are tidied, non-numeric cells and NaN rows are reported and dropped, unexpected columns are ignored, and a missing feature produces a readable error instead of a stack trace. With a `Target` column you get full metrics; without one the app switches to prediction-only mode. Only test data is uploaded, per the brief. A **Download bundled test_data.csv** button provides a template. |
| **b. Model selection dropdown** | Sidebar → *2 · Model* → **Choose a model** | All six trained pipelines, with a one-line description of the selected one. Every chart and table on the page follows the selection. |
| **c. Display of evaluation metrics** | Tab **Model evaluation** and tab **Compare all models** | The six required metrics as cards, each showing the delta against the offline training run (so the deployed artefact is provably the evaluated one), plus a full six-model comparison table with a colour gradient and a sortable bar chart per metric. |
| **d. Confusion matrix / classification report** | Tab **Model evaluation** | Confusion-matrix heatmap (raw counts or row-normalised), per-class precision / recall / F1 / support table, the plain-text `classification_report`, and one-vs-rest ROC curves with per-class AUC. |

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
pickled with (1.9.0) — because dependency drift is the most common cause
of a failed Streamlit deployment.

---

## Verification

`tests/test_project.py` runs 37 automated checks with `python -m pytest tests -q`:

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
Machine Learning Assignment 2 · executed on **BITS Virtual Lab** · submitted 18 August 2026
