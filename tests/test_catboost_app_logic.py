import numpy as np
import pandas as pd
import pytest

from aegis_catboost_logic import score_input_catboost


class FakeCatBoostModel:
    classes_ = np.array(["BenignPositive", "FalsePositive", "TruePositive"])
    feature_names_ = [
        "AegisSourceRowCount",
        "AegisUniqueAlertCount",
        "AegisUniqueDetectorCount",
        "DetectorId",
        "Category",
    ]

    def __init__(self):
        self.last_X = None

    def predict_proba(self, X):
        self.last_X = X.copy().reset_index(drop=True)
        # Valid deterministic probabilities.
        return np.tile(np.array([[0.2, 0.3, 0.5]], dtype=float), (len(X), 1))


def metadata_fixture():
    features = [
        "AegisSourceRowCount",
        "AegisUniqueAlertCount",
        "AegisUniqueDetectorCount",
        "DetectorId",
        "Category",
    ]
    return {
        "model_family": "catboost_row_level",
        "group_col": "IncidentKey",
        "class_order": ["BenignPositive", "FalsePositive", "TruePositive"],
        "numeric_features": features[:3],
        "categorical_features": features[3:],
        "feature_columns": features,
    }


def test_candidate_g_scores_one_row_per_incident_and_calculates_counts():
    raw = pd.DataFrame({
        "OrgId": [1, 1, 1, 1, 1, 1, 2],
        "IncidentId": [10, 10, 10, 10, 10, 20, 10],
        "AlertId": [100, 101, 101, 102, 103, 200, 300],
        "DetectorId": [1, 1, 2, 2, 3, 4, 5],
        "Category": ["A", "A", "B", None, "C", "D", "E"],
    })
    model = FakeCatBoostModel()
    scored, group = score_input_catboost(raw, model, metadata_fixture())

    assert group == "IncidentKey"
    assert len(scored) == 3
    assert scored[group].nunique() == 3
    incident = scored.loc[scored[group] == "1::10"].iloc[0]
    assert incident["AegisSourceRowCount"] == 5
    assert incident["AegisUniqueAlertCount"] == 4
    assert incident["AegisUniqueDetectorCount"] == 3
    assert (scored["Model_Prediction"] == "TruePositive").all()
    assert np.allclose(scored["Model_Confidence"], 0.5)
    probability_cols = [f"Prob_{c}" for c in metadata_fixture()["class_order"]]
    assert np.allclose(scored[probability_cols].sum(axis=1), 1.0)


def test_candidate_g_rejects_missing_alert_id():
    raw = pd.DataFrame({
        "OrgId": [1], "IncidentId": [10], "DetectorId": [1], "Category": ["A"]
    })
    with pytest.raises(ValueError, match="AlertId"):
        score_input_catboost(raw, FakeCatBoostModel(), metadata_fixture())


def test_sampling_is_stable_when_unrelated_incidents_are_added():
    target = pd.DataFrame({
        "OrgId": [1] * 9,
        "IncidentId": [10] * 9,
        "AlertId": list(range(100, 109)),
        "DetectorId": list(range(1, 10)),
        "Category": [f"cat-{i}" for i in range(9)],
    })
    extra = pd.DataFrame({
        "OrgId": [99] * 5,
        "IncidentId": [999] * 5,
        "AlertId": list(range(900, 905)),
        "DetectorId": [90, 91, 92, 93, 94],
        "Category": [f"extra-{i}" for i in range(5)],
    })

    first_model = FakeCatBoostModel()
    score_input_catboost(target, first_model, metadata_fixture())
    first_rows = first_model.last_X["Category"].tolist()

    second_model = FakeCatBoostModel()
    score_input_catboost(
        pd.concat([extra, target], ignore_index=True),
        second_model,
        metadata_fixture(),
    )
    second_rows = second_model.last_X.loc[
        second_model.last_X["AegisSourceRowCount"] == 9, "Category"
    ].tolist()
    assert first_rows == second_rows
