# Heart Failure Prediction: Comparative ML Analysis

Machine Learning Assignment No. 2 (project-based). Predicts whether a patient has heart disease from clinical
measurements, comparing Logistic Regression, Random Forest and Gradient Boosting, then improving results with
hyperparameter tuning and ensembles (Voting, Stacking). The final model is served through a Streamlit app.

- **Dataset:** [Heart Failure Prediction (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction), file `heart.csv`, target `HeartDisease`
- **Live app:** _add your Streamlit link here_

## Workflow
1. Exploratory data analysis (structure, duplicates, impossible zero values in `Cholesterol` and `RestingBP`, outliers, distributions, correlations)
2. Leakage-free preprocessing with a scikit-learn `Pipeline` (impute, scale, one-hot encode) and a stratified 80/20 split
3. Three models trained and evaluated identically (5-fold stratified CV plus held-out test set)
4. Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix
5. Improvement: RandomizedSearchCV on the best baseline, Soft Voting and Stacking ensembles
6. Final model saved with joblib, reloaded and verified, then served through Streamlit

## Results
Copy your numbers from `results/baseline_comparison.csv` and `results/before_after_improvement.csv`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| Gradient Boosting | | | | | |

## Repository structure
```
├── heart_disease_prediction.ipynb   # EDA, modelling, tuning, ensembles
├── app.py                           # Streamlit app
├── heart_model.joblib               # trained final model
├── app_meta.joblib                  # feature lists and input ranges for the app
├── heart.csv                        # dataset (or download from the Kaggle link)
├── requirements.txt
└── results/                         # comparison tables and figures
```

## Run locally
```bash
pip install -r requirements.txt
pip install jupyter
jupyter notebook          # run heart_disease_prediction.ipynb (creates the .joblib files and results/)
streamlit run app.py
```

## Notes
- Train the model in the same environment that will load it; saved scikit-learn models depend on library versions (pinned in `requirements.txt`).
- Zeros in `Cholesterol` and `RestingBP` are treated as missing and imputed inside the pipeline, using training data only.
- The test set is small (about 184 rows), so small metric differences between models are within noise.
- Educational project; not a medical device.

## Author
_Your name, roll number, department, RBU Nagpur_
