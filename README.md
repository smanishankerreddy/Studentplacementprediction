# Student Placement Prediction System

A supervised machine learning project that predicts whether a student will be
placed based on academic performance and skill-based features.

## Overview

This project builds and compares three classification models — Logistic
Regression, Decision Tree, and Random Forest — to predict student placement
outcomes, then selects the best-performing model, analyzes which features
matter most, and generates a placement probability for each student.

## Project Structure

```
StudentPlacementPrediction/
│
├── dataset.csv                  # Student records (academic + skill features)
├── placement_prediction.ipynb   # Full ML pipeline, step by step
├── model.pkl                    # Saved trained model + scaler + encoders
└── requirements.txt             # Python dependencies
```

## Features Used

| Feature | Description |
|---|---|
| Gender | Categorical |
| CGPA | Cumulative GPA |
| Internships | Number of internships completed |
| Projects | Number of academic/personal projects |
| Certifications | Number of certifications earned |
| AptitudeTestScore | Score out of 100 |
| SoftSkillsRating | Rated 1–10 |
| ExtracurricularActivities | Yes / No |
| PlacementTraining | Yes / No |
| SSC_Marks / HSC_Marks | Prior academic marks |

Target column: `PlacementStatus` (Yes / No)

## Pipeline Steps

1. **Load data** from `dataset.csv`.
2. **Handle missing values** — numeric columns filled with the mean,
   categorical columns filled with the mode.
3. **Encode categorical columns** using `LabelEncoder`.
4. **Scale features** using `StandardScaler`.
5. **Split** into 80% training / 20% testing (stratified by target).
6. **Train and compare** Logistic Regression, Decision Tree, and Random Forest.
7. **Evaluate** each model with accuracy, a confusion matrix, and a
   classification report.
8. **Select the best model** automatically based on test accuracy.
9. **Feature importance** — extracted from the tree-based model, shows which
   factors most influence placement.
10. **Placement probability** — `predict_proba` gives each student a
    probability of being placed, not just a Yes/No label.
11. **Save** the trained model, scaler, and encoders together into `model.pkl`
    so they can be reused for scoring new students later.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook placement_prediction.ipynb
```

Run all cells in order. To use your own data, replace `dataset.csv` and
update `DATA_PATH` / `TARGET_COLUMN` near the top of the notebook to match
your file and target column name.

## Using the Saved Model

```python
import pickle

with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
scaler = saved["scaler"]
encoders = saved["encoders"]
feature_names = saved["feature_names"]

# new_student_df must have the same columns as feature_names,
# with categorical columns encoded using the matching encoder in `encoders`
scaled = scaler.transform(new_student_df[feature_names])
prediction = model.predict(scaled)
probability = model.predict_proba(scaled)[:, 1]
```

## Note on the Dataset

`dataset.csv` included here is **synthetically generated** (10,000 rows) for
demonstration, since no real dataset was provided. Swap in your actual data
(keeping the same column names, or adjusting the notebook's config
variables) to get results that reflect real placement outcomes — reported
accuracy will vary depending on how predictive the real features are.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (Logistic Regression, Decision Tree, Random Forest,
  preprocessing, metrics)
