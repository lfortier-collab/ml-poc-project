# Student Performance Prediction — ML Proof of Concept

This project predicts whether a student will pass or fail based on behavioral, family, and academic features from the UCI Student Performance Dataset.
Three classifiers are compared — Logistic Regression, Random Forest, and Gradient Boosting — optimised on macro F1-score to handle the 78/22 class imbalance.
An interactive Streamlit app presents the analysis, model evaluation, SHAP explanations, and a live prediction demo with student profiling.

---

> This repository was built on top of the Albert School ML project template. The template defines the project structure and the main execution workflow.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Evaluate models and launch the app
python scripts/main.py
```

The app will be available at `http://localhost:8501`.

## Getting the Data

This project uses the [Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) from the UCI Machine Learning Repository.

1. Download the dataset from the link above and extract the archive.
2. Place `student-mat.csv` and `student-por.csv` into `data/raw/`.
3. Run the preprocessing notebook to generate the processed splits:

```bash
jupyter nbconvert --to notebook --execute notebooks/preprocessing.ipynb
```

This will create the following files in `data/processed/`:

| File | Description |
|---|---|
| `student_train.csv` | Scaled features for Logistic Regression (train) |
| `student_test.csv` | Scaled features for Logistic Regression (test) |
| `student_train_raw.csv` | Unscaled features for Random Forest & Gradient Boosting (train) |
| `student_test_raw.csv` | Unscaled features for Random Forest & Gradient Boosting (test) |
