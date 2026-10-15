# ORION TRIAGE

**Machine-learning-powered security incident classification and triage.**

ORION TRIAGE is a human-supervised Security Operations Centre (SOC)
incident-triage prototype built using the Microsoft GUIDE dataset.

## Capabilities

- Classifies incidents as Benign Positive, False Positive, or True Positive.
- Shows the predicted class and model confidence.
- Routes lower-confidence predictions for analyst review.
- Provides incident queue, model health, and analyst feedback audit views.
- Accepts compatible CSV input through the Streamlit interface.

## Technology

Python, pandas, NumPy, scikit-learn, Streamlit, joblib, and SHAP.

## Model versions and evaluation

The repository's original **Random Forest** remains the fallback model. Its bundled metadata records the following held-out scores:

| Metric | Random Forest result |
|---|---:|
| Accuracy | 68.45% |
| Macro F1 | 64.23% |
| Macro precision | 66.40% |
| Macro recall | 63.10% |
| Macro ROC AUC (one-vs-rest) | 81.78% |
| Test incidents | 2,000 |

This branch adds optional **CatBoost Candidate G** support. When its local `.cbm` artifact and metadata are available and load successfully, the app prefers Candidate G. Otherwise it keeps using Random Forest. Candidate G was evaluated on 5,918 held-out incidents under its stable per-incident sampling policy: accuracy 81.02%, macro F1 0.7799, macro precision 82.45%, macro recall 0.7577, and macro ROC-AUC (OvR) 92.89%. These Candidate G figures belong to that specific model and sampling policy, not the fallback Random Forest.

See [Candidate G integration instructions](CANDIDATE_G_INTEGRATION.md) for model placement, optional dependency, testing, sampling policy, and SHAP limitations.

## Run locally

1. Install a compatible Python version.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the interface:

   ```bash
   streamlit run app.py
   ```

5. Upload a compatible Microsoft GUIDE-derived CSV file.

Raw datasets and local feedback databases are not included. Input data
must contain the fields expected by the application's feature-processing
code. Consult the source code before using a different CSV schema.

## Tests

Install the development dependencies and run:

```bash
pip install -r requirements-dev.txt
pytest -q tests
```

## Responsible use and limitations

This is an academic and portfolio prototype, not a production SOC system.
Predictions are decision-support suggestions, not confirmed ground truth.
Analysts should review the underlying evidence before acting. The saved
evaluation results describe a particular held-out test set and do not
guarantee performance on other environments or datasets.

## Project structure

- `app.py` — Streamlit interface
- `aegis_app_logic.py` — scoring and analyst-feedback logic
- `aegis_features.py` — incident feature preparation
- `train_aegis_ir.py` — training and evaluation code
- `prepare_guide_sample.py` — dataset preparation utility
- `models/` — model artifact and metadata
- `tests/` — automated tests
- `reports/plots/` — existing evaluation figures
