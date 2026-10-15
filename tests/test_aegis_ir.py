import sqlite3

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aegis_features import (
    GROUP_COL,
    aggregate_incident_features,
    build_incident_training_table,
    ensure_group_key,
    validate_incident_labels,
)
from aegis_app_logic import initialize_feedback_db, read_feedback, save_feedback, score_input


def fixture_rows():
    return pd.DataFrame({
        "OrgId": [1, 1, 2, 1, 1, 2, 3, 3, 4],
        "IncidentId": [10, 10, 10, 20, 20, 20, 30, 30, 40],
        "IncidentGrade": [
            "TruePositive", "TruePositive", "FalsePositive",
            "BenignPositive", "BenignPositive", "FalsePositive",
            "TruePositive", "TruePositive", "BenignPositive",
        ],
        "AlertId": [1001, 1002, 1003, 2001, 2002, 2003, 3001, 3001, 4001],
        "DetectorId": [11, 12, 13, 21, 22, 23, 31, 32, 41],
        "AlertCount": [2, 4, 1, 2, 6, 4, 3, 5, 7],
        "Category": ["Execution", "Persistence", "InitialAccess", "Discovery", "Discovery", "Execution", "Execution", "Execution", "Discovery"],
        "LastVerdict": ["leak"] * 9,
        "ActionGrouped": ["leak"] * 9,
    })


def test_composite_key_disambiguates_incident_ids_across_orgs():
    df = ensure_group_key(fixture_rows())
    assert GROUP_COL in df.columns
    assert df.loc[0, GROUP_COL] != df.loc[2, GROUP_COL]
    assert df.loc[0, GROUP_COL] == "1::10"
    assert df.loc[2, GROUP_COL] == "2::10"


def test_incident_aggregation_is_one_row_per_org_incident():
    df = ensure_group_key(fixture_rows())
    incident_table, numeric, categorical, excluded = build_incident_training_table(df)
    assert len(incident_table) == 6
    assert incident_table[GROUP_COL].nunique() == 6
    assert "LastVerdict" in excluded
    row = incident_table.loc[incident_table[GROUP_COL] == "1::10"].iloc[0]
    assert row["AlertCount"] == 3.0
    # Tie in categorical mode is resolved deterministically in lexical order.
    assert row["Category"] == "Execution"


def test_conflicting_labels_for_same_composite_incident_fail_closed():
    df = fixture_rows().iloc[[0, 1]].copy()
    df.loc[df.index[1], "IncidentGrade"] = "FalsePositive"
    with pytest.raises(ValueError, match="conflicting"):
        validate_incident_labels(df)


def fitted_test_pipeline(df):
    work = ensure_group_key(df)
    table, numeric, categorical, _ = build_incident_training_table(work)
    features = numeric + categorical
    X = table[features]
    y = table["IncidentGrade"]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    model = Pipeline([("pre", ColumnTransformer(transformers=transformers)), ("clf", LogisticRegression(max_iter=500))])
    model.fit(X, y)
    metadata = {
        "group_col": GROUP_COL,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_columns": features,
        "class_order": [str(value) for value in model.classes_],
        "winning_model_name": "test-fixture",
    }
    return model, metadata


def test_score_input_returns_one_row_per_composite_incident():
    df = fixture_rows()
    model, metadata = fitted_test_pipeline(df)
    scored, group_col = score_input(df, model, metadata)
    assert group_col == GROUP_COL
    assert len(scored) == 6
    assert scored[group_col].nunique() == 6
    assert np.allclose(scored[[f"Prob_{c}" for c in metadata["class_order"]]].sum(axis=1), 1.0)


def test_sqlite_migration_and_feedback_roundtrip(tmp_path):
    db_path = tmp_path / "feedback.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE analyst_feedback (
            timestamp TEXT, incident_id TEXT, model_prediction TEXT,
            model_confidence REAL, analyst_verdict TEXT, analyst_notes TEXT,
            review_duration_sec INTEGER
        )""")
        conn.execute("INSERT INTO analyst_feedback VALUES (CURRENT_TIMESTAMP, 'old', 'FalsePositive', 0.7, 'FalsePositive', '', 3)")
    initialize_feedback_db(db_path)
    save_feedback(db_path, "new", "test-model-v1", "TruePositive", 0.8,
                  {"TruePositive": 0.8, "BenignPositive": 0.1, "FalsePositive": 0.1},
                  "TruePositive", "verified", "fixture.csv")
    results = read_feedback(db_path, limit=10)
    assert len(results) == 2
    assert "model_version" in results.columns
    assert "new" in set(results["incident_id"])
