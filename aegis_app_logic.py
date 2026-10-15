"""Testable application logic separated from Streamlit rendering."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis_features import GROUP_COL, aggregate_incident_features, ensure_group_key


def score_input(raw_df: pd.DataFrame, pipeline: Any, metadata: dict) -> tuple[pd.DataFrame, str]:
    """Validate schema, aggregate raw rows by incident, and return scored incidents."""
    if raw_df.empty:
        raise ValueError("The input CSV has no data rows.")
    group_col = metadata.get("group_col", GROUP_COL)
    raw_df = ensure_group_key(raw_df, group_col)
    numeric = metadata["numeric_features"]
    categorical = metadata["categorical_features"]
    feature_cols = metadata.get("feature_columns", numeric + categorical)

    if group_col not in raw_df.columns:
        raise ValueError(f"Required grouping column '{group_col}' is missing.")
    engineered_cols = {
        "AegisSourceRowCount",
        "AegisUniqueAlertCount",
        "AegisUniqueDetectorCount",
    }
    raw_feature_cols = [
        col for col in feature_cols if col not in engineered_cols
    ]
    missing = [col for col in raw_feature_cols if col not in raw_df.columns]
    if missing:
        preview = ", ".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        raise ValueError(f"CSV is missing required model features: {preview}{suffix}")
    if list(map(str, pipeline.classes_)) != list(map(str, metadata["class_order"])):
        raise ValueError("Model classes and metadata class order do not match. Retrain or restore matching artifacts.")

    incident_features = aggregate_incident_features(
        raw_df,
        feature_columns=feature_cols,
        numeric_features=numeric,
        categorical_features=categorical,
        group_col=group_col,
    )
    X = incident_features[feature_cols]
    probabilities = pipeline.predict_proba(X)
    predictions = pipeline.predict(X)

    scored = incident_features.copy()
    display_cols = [col for col in ("OrgId", "IncidentId") if col in raw_df.columns]
    if display_cols:
        display = raw_df[[group_col] + display_cols].drop_duplicates(group_col, keep="first")
        scored = scored.merge(display, on=group_col, how="left", validate="one_to_one")
    scored["Model_Prediction"] = predictions.astype(str)
    scored["Model_Confidence"] = probabilities.max(axis=1)
    for index, class_name in enumerate(metadata["class_order"]):
        scored[f"Prob_{class_name}"] = probabilities[:, index]
    return scored, group_col


def initialize_feedback_db(db_path: str | Path) -> None:
    """Create or migrate the local feedback table, including prior prototype schemas."""
    with closing(sqlite3.connect(str(db_path))) as conn:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyst_feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    incident_id TEXT NOT NULL,
                    model_version TEXT,
                    model_prediction TEXT NOT NULL,
                    model_confidence REAL NOT NULL,
                    predicted_probabilities TEXT,
                    analyst_verdict TEXT NOT NULL,
                    analyst_notes TEXT,
                    source_name TEXT
                )
            """)
            existing = {
                row[1] for row in conn.execute("PRAGMA table_info(analyst_feedback)").fetchall()
            }
            # New columns are nullable when added to a populated legacy table.
            migrations = {
                "timestamp": "TEXT",
                "incident_id": "TEXT",
                "model_version": "TEXT",
                "model_prediction": "TEXT",
                "model_confidence": "REAL",
                "predicted_probabilities": "TEXT",
                "analyst_verdict": "TEXT",
                "analyst_notes": "TEXT",
                "source_name": "TEXT",
            }
            for column, sql_type in migrations.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE analyst_feedback ADD COLUMN {column} {sql_type}")
            conn.execute(
                "UPDATE analyst_feedback SET timestamp = CURRENT_TIMESTAMP WHERE timestamp IS NULL"
            )


def save_feedback(
    db_path: str | Path,
    incident_id: str,
    model_version: str,
    prediction: str,
    confidence: float,
    probabilities: dict[str, float],
    verdict: str,
    notes: str,
    source_name: str,
) -> None:
    with closing(sqlite3.connect(str(db_path))) as conn:
        with conn:
            conn.execute("""
                INSERT INTO analyst_feedback
                (incident_id, model_version, model_prediction, model_confidence,
                 predicted_probabilities, analyst_verdict, analyst_notes, source_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(incident_id), str(model_version), str(prediction), float(confidence),
                json.dumps(probabilities), str(verdict), str(notes), str(source_name),
            ))


def read_feedback(db_path: str | Path, limit: int = 100) -> pd.DataFrame:
    with closing(sqlite3.connect(str(db_path))) as conn:
        return pd.read_sql_query(
            "SELECT rowid AS audit_row, timestamp, incident_id, model_version, "
            "model_prediction, model_confidence, analyst_verdict, analyst_notes, source_name "
            "FROM analyst_feedback ORDER BY timestamp DESC, rowid DESC LIMIT ?",
            conn, params=(int(limit),),
        )
