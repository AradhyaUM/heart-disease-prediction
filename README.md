# Heart Disease Prediction

Machine Learning Assignment No. 2.
**Author:** M ARADHYA 
**GitHub:** https://github.com/AradhyaUM/heart-disease-prediction

> Educational project. This is not a medical device and not a substitute for professional diagnosis.

## Problem

Predict whether a patient has heart disease (binary target `HeartDisease`) from 11 clinical features, and compare several machine-learning models trained and evaluated the same way.

## Dataset

Kaggle: **Heart Failure Prediction Dataset** by fedesoriano
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

918 rows, 11 features (Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope), target `HeartDisease`. The file is included as `heart.csv`.

## Repository contents

| File / folder | Purpose |
|---|---|
| `heart_disease_prediction.ipynb` | Full notebook: EDA, preprocessing, models, evaluation, tuning, ensembles |
| `heart.csv` | Dataset |
| `app.py` | Streamlit app (Predict tab and Model results tab) |
| `heart_model.joblib`, `app_meta.joblib` | Saved final model and app metadata |
| `results/` | Charts (PNG) and result tables (`baseline_comparison.csv`, `before_after_improvement.csv`) |
| `requirements.txt` | Pinned Python dependencies |

## Method

- Models: Logistic Regression, Random Forest, Gradient Boosting (scikit-learn), all trained and evaluated on the same held-out test set.
- Metrics: accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC.
- Improvement: RandomizedSearchCV on the best baseline model, plus Soft Voting and Stacking ensembles, compared before vs after.

## Results

Evaluated on a held-out test set of 184 patients (102 with heart disease, 82 without).

### Baseline models

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8859 | 0.8857 | 0.9118 | 0.8986 | 0.9329 |
| Random Forest | 0.8804 | 0.8846 | 0.9020 | 0.8932 | 0.9294 |
| Gradient Boosting | 0.8750 | 0.8911 | 0.8824 | 0.8867 | 0.9173 |

### Before vs after tuning / ensembling

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.8859 | 0.8857 | 0.9118 | 0.8986 | 0.9329 |
| Logistic Regression (tuned) | 0.8859 | 0.8857 | 0.9118 | 0.8986 | 0.9335 |
| Voting (soft) | 0.8967 | 0.9109 | 0.9020 | 0.9064 | 0.9357 |
| Stacking | 0.8913 | 0.9020 | 0.9020 | 0.9020 | 0.9358 |

**Final model in the app:** Voting (soft), selected for the highest test F1 (0.9064).

### How to read these numbers

- The test set has only 184 rows, so one patient is about 0.5 percentage points of accuracy. Voting gets 165 of 184 correct and baseline Logistic Regression gets 163.
- A rough 95% interval for an accuracy near 0.89 on 184 rows is about plus or minus 4.5 points, so the differences between the models are within noise. All models perform similarly (accuracy 0.875 to 0.897, ROC-AUC 0.917 to 0.936).
- Tuning Logistic Regression left every metric unchanged except ROC-AUC (0.9329 to 0.9335).
- The final model was chosen using test-set F1, so its test score is slightly optimistic.

## Run locally

Python 3.12 is recommended (the pinned package versions do not support the newest Python releases).

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. Do not run the app with `python app.py`.

To retrain, run all cells of `heart_disease_prediction.ipynb` in the same environment, which regenerates the `.joblib` files and the `results/` folder. The model must be loaded with the same scikit-learn version it was trained with (1.6.1).

## Limitations

- Small dataset (918 patients) and small test set, so metric differences between models are not statistically meaningful.
- Single train/test split; results could shift with a different split.
- Educational use only.
