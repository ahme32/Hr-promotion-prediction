# HR Analytics — Employee Promotion Prediction

**Live Demo:** https://hr-promotion-prediction-zedhrptm5zpkhppp5xmv3a.streamlit.app/

A Data Science project that predicts whether an employee will be promoted, using their
performance and profile data. Built as a final project for **Epsilon AI**.

> Completed as a final project for the **Epsilon AI** Data Science program.
> Epsilon AI Academy: https://eg.epsilonaiglobal.com/academy

---

## Problem
**Binary Classification** — predict the target `is_promoted` (1 = promoted, 0 = not promoted).

The dataset is imbalanced (only ~8.5% of employees were promoted), so the models are judged on
**precision and recall**, not accuracy.

## Dataset
- Source: HR Analytics dataset (Analytics Vidhya "WNS" Hackathon 2018), from Kaggle — **not** from the Epsilon list.
- Size: 54,808 rows × 13 columns (mixed numeric and categorical).
- It is a *dirty* dataset: missing values in `education` and `previous_year_rating`, duplicate rows,
  and a column name with a special character.

## Project steps (in the notebook)
1. **Project Understanding** — business problem and questions.
2. **Data Cleaning** — handle missing values, drop duplicates, fix column names.
3. **EDA** — univariate and bivariate analysis, 6 kinds of plots, plus a t-test.
4. **Feature Engineering** — new features `total_score` and `high_training_score`.
5. **Feature Selection** — Embedded method (Random Forest feature importance).
6. **Modeling** — Logistic Regression, KNN, Decision Tree, Random Forest, compared and tuned with GridSearchCV, validated with cross-validation.
7. **Deployment** — a Streamlit web app (`app.py`).

## Results
The best model is a **tuned Random Forest** (`max_depth=20`, `n_estimators=200`):
- Precision ≈ **0.41**, Recall ≈ **0.47** on the test set (both above the required 0.3).
- Cross-validation average F1 ≈ **0.42**.

## Files
| File | Description |
|---|---|
| `HR_Promotion_Prediction.ipynb` | The full notebook (all steps). |
| `app.py` | The Streamlit web app for predictions. |
| `train.csv` | The dataset. |
| `model.pkl` | The trained (tuned) Random Forest model. |
| `scaler.pkl` | The fitted StandardScaler. |
| `features.pkl` | The list of feature columns used by the model. |
| `requirements.txt` | The Python packages needed to run the project. |

## How to run the web app
```bash
pip install -r requirements.txt
streamlit run app.py
```
Make sure `model.pkl`, `scaler.pkl`, and `features.pkl` are in the same folder as `app.py`.
