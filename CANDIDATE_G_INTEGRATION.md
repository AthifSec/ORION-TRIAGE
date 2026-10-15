# ORION TRIAGE — Candidate G integration

This branch adds optional support for the row-level CatBoost Candidate G model. It does not include the 15.6 MB `.cbm` binary and does not replace the existing Random Forest model artifact.

## Install the optional dependency

Activate the project's virtual environment and install:

```bash
pip install -r requirements-catboost.txt
```

## Add the trained model locally

Download `AegisIR_CandidateG_Final_2026-10-15.cbm` from your Google Drive experiment output and copy it to:

`models/aegis_ir_candidate_g.cbm`

The matching metadata file is already in `models/catboost_candidate_g_metadata.json`. The binary is intentionally not committed. If Candidate G's model is missing or cannot load, the app falls back to the existing Random Forest pipeline.

## Run tests and launch

```bash
pytest -q tests
streamlit run app.py
```

The app shows the active model and version so you can confirm which model is running. Candidate G's class probabilities are averaged across up to four source rows per incident. The three count features use the complete incident queue. Categorical nulls are converted to `__MISSING__`, matching training preprocessing.

## Stable sampling policy and metrics

The original training experiment selected up to four source rows by shuffling all incident groups with a shared random state. That procedure could select different rows for an incident when unrelated incidents were added or removed from the queue. Production inference now derives a deterministic SHA-256 seed from each incident key; a dedicated automated test checks that this selection remains stable.

The same 5,918 held-out incidents were re-evaluated under the production sampling policy. These figures are associated with that policy; the original global-sampling evaluation also reproduced exactly before switching the deployment sampler.

| Metric | Stable-policy held-out result |
|---|---:|
| Accuracy | 81.02% |
| Macro F1 | 0.7799 |
| Macro precision | 82.45% |
| Macro recall | 0.7577 |
| Macro ROC-AUC (one-vs-rest) | 92.89% |
| Test incidents | 5,918 |

The same evaluation split was used to compare sampling policies; treat these metrics as held-out validation of this implementation, not as a guarantee of future SOC performance. TruePositive recall remains a separate important measure to watch.

## SHAP limitation

The existing local SHAP explanation targets the old aggregated Random Forest input. It is disabled for Candidate G until a row-level explanation method that matches probability averaging is implemented. Analyst review and audit logging remain human-supervised; the app does not take automatic response actions.
