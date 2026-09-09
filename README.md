# Heart Disease Prediction

A compact machine-learning walkthrough that trains a **logistic regression** classifier to predict whether a patient record is labelled as having heart disease. The complete workflow—from loading the CSV through evaluating the model and predicting one example—is contained in the accompanying Jupyter notebook.

> **Medical disclaimer:** This project is an educational demonstration, not a diagnostic tool. A model trained on this small sample dataset must not be used to make medical decisions. Seek advice from a qualified healthcare professional for any health concern.

## Contents

```text
.
├── heart_disease_prediction.ipynb   # End-to-end exploratory analysis and model workflow
├── sample_data/
│   └── heart_disease_data.csv       # Input dataset used by the notebook
└── README.md
```

## What the notebook does

1. Imports NumPy, pandas, and scikit-learn utilities.
2. Loads `sample_data/heart_disease_data.csv` into a pandas `DataFrame`.
3. Inspects the data with previews, shape, missing-value counts, data types, descriptive statistics, and target distribution.
4. Separates the `target` column from the 13 predictor columns.
5. Creates a stratified 80/20 train/test split using `random_state=2`.
6. Fits `sklearn.linear_model.LogisticRegression(max_iter=1000)` on the training partition.
7. Reports accuracy for both the training and held-out test partitions.
8. Predicts the class for a single row selected from the feature data.

## Dataset

The bundled CSV contains **303 patient records**, **13 input features**, and one binary `target` label (14 columns in total). The notebook's saved inspection output reports no missing values. Its target convention is:

| Target value | Meaning used by this project |
| --- | --- |
| `1` | Defective heart / heart-disease label |
| `0` | Healthy / no-heart-disease label |

The saved target counts are 165 records labelled `1` and 138 labelled `0`. Feature names and conventional meanings are listed below. Categorical values are stored numerically; retain these encodings when preparing data for the notebook.

| Column | Description | Typical encoding / unit |
| --- | --- | --- |
| `age` | Age | Years |
| `sex` | Biological sex | `0` = female, `1` = male |
| `cp` | Chest-pain type | Categorical code `0`–`3` |
| `trestbps` | Resting blood pressure | mm Hg |
| `chol` | Serum cholesterol | mg/dL |
| `fbs` | Fasting blood sugar above 120 mg/dL | `0` = no, `1` = yes |
| `restecg` | Resting electrocardiographic result | Categorical code `0`–`2` |
| `thalach` | Maximum heart rate achieved | Beats per minute |
| `exang` | Exercise-induced angina | `0` = no, `1` = yes |
| `oldpeak` | ST depression induced by exercise relative to rest | Numeric value |
| `slope` | Slope of the peak exercise ST segment | Categorical code `0`–`2` |
| `ca` | Number of major vessels colored by fluoroscopy | Integer code `0`–`4` |
| `thal` | Thalassemia test result | Categorical code in the supplied data |
| `target` | Classification label | `0` or `1`; prediction target |

## Requirements

- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- scikit-learn

Install the Python dependencies in an isolated virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyter numpy pandas scikit-learn
```

## Run the project

From the repository root, start Jupyter:

```bash
jupyter notebook
```

Open `heart_disease_prediction.ipynb` and run cells from top to bottom. The relative data path in the notebook is `./sample_data/heart_disease_data.csv`, so launching Jupyter from the repository root ensures that the CSV is found.

For a non-interactive execution after installing the requirements, use:

```bash
jupyter nbconvert --to notebook --execute --inplace heart_disease_prediction.ipynb
```

## Model and evaluation

The notebook uses a logistic-regression baseline with these reproducibility-relevant choices:

| Setting | Value |
| --- | --- |
| Estimator | `LogisticRegression` |
| Maximum iterations | `1000` |
| Test-set fraction | `0.20` |
| Split seed | `2` |
| Class preservation | `stratify=Y` |
| Resulting split | 242 training rows / 61 test rows |
| Metric | `accuracy_score` |

The notebook's saved outputs show an accuracy of approximately **85.54%** on the training data and **80.33%** on the held-out test data. These values are specific to the included dataset, split, library behavior, and preprocessing shown in the notebook; they are not clinical-performance claims.

## Make a prediction with the trained model

The final notebook cell demonstrates prediction for one record. For a new record, provide every feature in exactly the same order as `X.columns` and preserve the training encodings. Using a one-row `DataFrame` retains column names and avoids the feature-name warning that can occur when passing a bare NumPy array:

```python
new_patient = pd.DataFrame(
    [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]],
    columns=X.columns,
)

prediction = model.predict(new_patient)[0]
print("Heart-disease label" if prediction == 1 else "No-heart-disease label")
```

This only produces a model label. It does not explain a diagnosis, estimate treatment needs, or replace clinical evaluation.

## Notes and limitations

- The dataset is small, so a single train/test split can give a noisy estimate of generalization performance.
- The workflow does not perform feature scaling, cross-validation, calibration, fairness analysis, hyperparameter search, or external validation.
- Several features are categorical codes. Their meanings and acceptable values should be validated against the data source before collecting new records.
- Do not treat the target label or the displayed accuracy as evidence of clinical safety or effectiveness.

## Suggested next steps

- Add a `requirements.txt` or locked environment file for repeatable installations.
- Use cross-validation and report precision, recall, F1 score, ROC-AUC, and a confusion matrix in addition to accuracy.
- Build a preprocessing pipeline that explicitly handles numeric and categorical fields.
- Persist the trained pipeline and add schema validation before accepting new inputs.
- Evaluate performance on appropriately governed external data before considering any real-world use.

## License

No license file is currently included. Add an explicit license before redistributing or reusing the project beyond the terms that apply to its original source and data.
