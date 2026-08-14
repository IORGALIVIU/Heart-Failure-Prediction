# Heart Failure Prediction

Predicts the likelihood of death due to heart failure using clinical data and supervised machine learning. Cardiovascular diseases are the leading cause of death globally, claiming an estimated 17.9 million lives each year. Early detection through predictive modeling can support timely medical intervention.

## Dataset

- **Source**: [Kaggle — Heart Failure Prediction Dataset](https://www.kaggle.com/code/karnikakapoor/heart-failure-prediction-ann/input)
- **Features**: `age`, `anaemia`, `creatinine_phosphokinase`, `diabetes`, `ejection_fraction`, `high_blood_pressure`, `platelets`, `serum_creatinine`, `serum_sodium`, `sex`, `smoking`
- **Target**: `DEATH_EVENT` (binary classification)

> **Note on `time`:** the original dataset also includes a `time` column (days of follow-up until death or end of study). It was excluded from this project's feature set — `time` is only known *after* the outcome has occurred and is not available for a new patient at prediction time. Including it produces an inflated, non-deployable accuracy (~84%) driven almost entirely by this single leaking feature. All results below reflect training **without** `time`.

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn — `LogisticRegression`, `SVC`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `GaussianNB`
- Model selection: `train_test_split`, `cross_val_score` (5-fold), `GridSearchCV`

## Results

Seven models were trained and compared on a held-out test set (30%, ~90 patients):

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Support Vector Machine | 71.1% | 57.7% | 50.0% | 53.6% |
| K-Nearest Neighbors | 68.9% | 55.6% | 33.3% | 41.7% |
| Logistic Regression | 71.1% | 59.1% | 43.3% | 50.0% |
| Decision Tree | 64.4% | 45.5% | 33.3% | 38.5% |
| Random Forest | 74.4% | 64.0% | 53.3% | 58.2% |
| **Gradient Boosting** | **73.3%** | 59.4% | **63.3%** | **61.3%** |
| Gaussian Naive Bayes | 70.0% | 57.1% | 40.0% | 47.1% |
| Logistic Regression (Tuned) | 70.0% | 56.5% | 43.3% | 49.1% |

**Gradient Boosting** offers the best overall balance, with the highest F1-score and the best recall (63.3%) among all models — the most clinically relevant metric here, since missing a genuinely high-risk patient (a false negative) is the costliest type of error in this setting. Random Forest is a close second on accuracy and F1.

Hyperparameter tuning on Logistic Regression via `GridSearchCV` (5-fold CV, optimized for F1) selected a heavily regularized model (`C=0.001`) but did not outperform the untuned baseline on the test set — consistent with the added variance of tuning on a small dataset (~299 records).

The most important predictors identified (from the tuned Logistic Regression's coefficients) were `age`, `serum_creatinine`, and `ejection_fraction` — clinically meaningful variables that, unlike `time`, are genuinely available at the point of care.

## Key Takeaways

- **A high score is a reason to investigate, not celebrate.** The original 84% accuracy run (with `time` included) looked strong but wasn't deployable — it depended on information only available in hindsight.
- **Data leakage isn't always a coding bug.** `time` was a statistically valid, well-behaved feature; the problem was that it couldn't exist at prediction time for a new patient.
- **Recall matters more than accuracy here.** In a clinical risk-assessment context, false negatives (missed high-risk patients) are far more costly than false positives.
- **Without `time`, no model clears ~74% accuracy or ~63% recall** — this is the honest performance ceiling on this dataset with these features, and a more realistic starting point for further work (e.g. class-imbalance handling, additional feature engineering, or a larger dataset).

## How to Run

1. Clone the repository or download the project folder.
2. Install required libraries:
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
```
3. Open `heart_failure.ipynb` in Jupyter Notebook and run all cells.
