"""Inference adapter for ORION TRIAGE Candidate G (row-level CatBoost)."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

ENGINEERED_COUNT_FEATURES = (
    "AegisSourceRowCount",
    "AegisUniqueAlertCount",
    "AegisUniqueDetectorCount",
)
MISSING_TOKEN = "__MISSING__"
MAX_SOURCE_ROWS_PER_INCIDENT = 4
ROW_SAMPLE_SEED = 73


def score_input_catboost(
    raw_df: pd.DataFrame,
    model: Any,
    metadata: dict,
) -> tuple[pd.DataFrame, str]:
    """Score selected source rows, then average their class probabilities per incident.

    Incident counts are computed over the complete input. Up to four categorical
    source rows are selected using a deterministic per-incident seed so unrelated
    incidents cannot change which evidence rows are scored for a given incident.
    """
    if raw_df.empty:
        raise ValueError("The input CSV has no data rows.")

    group_col = metadata.get("group_col", "IncidentKey")
    numeric = list(metadata["numeric_features"])
    categorical = list(metadata["categorical_features"])
    feature_cols = list(metadata.get("feature_columns", numeric + categorical))
    class_order = [str(label) for label in metadata["class_order"]]

    missing_engineered = [c for c in ENGINEERED_COUNT_FEATURES if c not in numeric]
    if missing_engineered:
        raise ValueError(
            "Candidate G metadata must include all engineered count features; "
            f"missing: {missing_engineered}"
        )
    if feature_cols != numeric + categorical:
        raise ValueError(
            "Candidate G feature order must be numeric features followed by categorical features."
        )

    model_classes = [str(label) for label in getattr(model, "classes_", [])]
    if not model_classes:
        raise ValueError("Loaded model does not expose class labels.")
    if set(model_classes) != set(class_order):
        raise ValueError("Model classes and Candidate G metadata do not match.")
    model_features = getattr(model, "feature_names_", None)
    if model_features and list(model_features) != feature_cols:
        raise ValueError("Loaded Candidate G feature order does not match its metadata.")

    raw = raw_df.copy()
    if group_col not in raw.columns:
        if group_col == "IncidentKey" and {"OrgId", "IncidentId"}.issubset(raw.columns):
            def normalize_id(series: pd.Series) -> pd.Series:
                return (
                    series.astype("string")
                    .str.strip()
                    .str.replace(r"\.0$", "", regex=True)
                )

            raw[group_col] = (
                normalize_id(raw["OrgId"]) + "::" + normalize_id(raw["IncidentId"])
            )
        else:
            raise ValueError(
                f"Cannot build required grouping key '{group_col}'. Provide that column "
                "or provide OrgId and IncidentId for IncidentKey."
            )

    if raw[group_col].isna().any():
        raise ValueError(f"Invalid or missing incident grouping key: {group_col}")
    raw[group_col] = raw[group_col].astype("string").str.strip()
    if raw[group_col].isna().any() or (raw[group_col] == "").any():
        raise ValueError(f"'{group_col}' contains empty grouping keys.")
    raw[group_col] = raw[group_col].astype(str)

    required = list(dict.fromkeys(categorical + ["AlertId", "DetectorId"]))
    missing = [column for column in required if column not in raw.columns]
    if missing:
        preview = ", ".join(missing[:20])
        raise ValueError(f"CSV is missing required Candidate G source fields: {preview}")

    incident_order = raw[group_col].drop_duplicates().astype(str).tolist()
    grouped = raw.groupby(group_col, sort=False, dropna=False)
    incident_features = pd.DataFrame({group_col: incident_order})

    count_map = {
        "AegisSourceRowCount": grouped.size(),
        "AegisUniqueAlertCount": grouped["AlertId"].nunique(dropna=True),
        "AegisUniqueDetectorCount": grouped["DetectorId"].nunique(dropna=True),
    }
    for column in ENGINEERED_COUNT_FEATURES:
        incident_features[column] = (
            incident_features[group_col].map(count_map[column]).astype(np.float32)
        )

    def stable_mode(series: pd.Series):
        values = series.dropna().astype("string")
        if values.empty:
            return pd.NA
        counts = values.value_counts(dropna=True)
        winners = sorted(str(value) for value in counts[counts == counts.max()].index.tolist())
        return winners[0]

    for column in categorical:
        mode_map = grouped[column].agg(stable_mode)
        incident_features[column] = incident_features[group_col].map(mode_map)
    incident_features = incident_features[[group_col] + feature_cols]

    def incident_seed(incident_key: str) -> int:
        seed_text = f"ORION-TRIAGE-Candidate-G-{ROW_SAMPLE_SEED}::{incident_key}"
        return int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)

    sampled_groups = []
    for incident_key, group in raw.groupby(group_col, sort=False, dropna=False):
        sampled_groups.append(
            group.sample(frac=1, random_state=incident_seed(str(incident_key)))
            .head(MAX_SOURCE_ROWS_PER_INCIDENT)
        )
    sampled = pd.concat(sampled_groups, ignore_index=True)

    count_summary = incident_features[[group_col] + numeric]
    sampled = sampled[[group_col] + categorical].merge(
        count_summary,
        on=group_col,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    for column in categorical:
        sampled[column] = (
            sampled[column].astype("string").fillna(MISSING_TOKEN).astype(str)
        )
    for column in numeric:
        sampled[column] = (
            pd.to_numeric(sampled[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .astype(np.float32)
        )

    sampled = sampled[[group_col] + feature_cols]
    probabilities = np.asarray(model.predict_proba(sampled[feature_cols]), dtype=float)
    if probabilities.shape != (len(sampled), len(model_classes)):
        raise ValueError("Candidate G returned a probability matrix with an unexpected shape.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Candidate G returned non-finite probabilities.")

    row_probs = pd.DataFrame(probabilities, columns=model_classes)
    row_probs[group_col] = sampled[group_col].to_numpy()
    incident_probs = row_probs.groupby(group_col, sort=False)[model_classes].mean()
    incident_probs = incident_probs.reindex(incident_order)[class_order]

    if incident_probs.isna().any().any():
        raise ValueError("At least one incident did not receive a probability prediction.")

    scored = incident_features.copy()
    for class_name in class_order:
        scored[f"Prob_{class_name}"] = incident_probs[class_name].to_numpy(dtype=float)

    probability_array = incident_probs.to_numpy(dtype=float)
    scored["Model_Prediction"] = np.asarray(class_order, dtype=object)[
        probability_array.argmax(axis=1)
    ]
    scored["Model_Confidence"] = probability_array.max(axis=1)

    display_cols = [column for column in ("OrgId", "IncidentId") if column in raw.columns]
    if display_cols:
        display = raw[[group_col] + display_cols].drop_duplicates(group_col, keep="first")
        scored = scored.merge(
            display, on=group_col, how="left", validate="one_to_one", sort=False
        )

    return scored, group_col
