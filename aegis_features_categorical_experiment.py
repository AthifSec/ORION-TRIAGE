"""Shared, deterministic feature preparation for ORION TRIAGE training and inference.

The model's prediction unit is one incident identified by the composite OrgId + IncidentId
key. GUIDE incident identifiers are organization-scoped, so grouping on IncidentId alone
can merge unrelated incidents across organizations. Multiple source rows are aggregated
with numeric medians and deterministic categorical modes; validate this baseline policy.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

TARGET_COL = "IncidentGrade"
GROUP_COL = "IncidentKey"

LEAKAGE_COLUMNS = {
    "lastverdict",
    "actiongrouped",
    "actiongranular",
    "remediationstate",
    "resolutiontime",
    "resolutionstatus",
    "closedby",
    "investigationresult",
}

# Conservative exclusions for identifiers or values that can act as near-unique keys.
NON_FEATURE_ID_COLUMNS = {
    "id",
    "incidentid",
    "incidentkey",
    "alertid",
    "orgid",
    "organizationid",
    "deviceid",
    "hostid",
    "entityid",
    "processid",
    "accountobjectid",
    "hash",
    "filehash",
    "sha1",
    "sha256",
    "ipaddress",
}

# These fields are often categorical IDs/codes, even if the CSV parses them as numbers.
FORCE_CATEGORICAL_COLUMNS = {
    "detectorid",
    "alerttitle",
    "category",
    "entitytype",
    "osfamily",
    "osplatform",
    "mitretechniques",
}


def _key(name: object) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "")


def ensure_group_key(frame: pd.DataFrame, group_col: str = GROUP_COL) -> pd.DataFrame:
    """Ensure a stable incident key exists without using it as a predictor.

    For GUIDE, IncidentId is organization-scoped. We therefore derive an explicit
    composite key from OrgId and IncidentId. If the requested key already exists
    (for example in a prepared sample), it is validated and preserved.
    """
    if group_col in frame.columns:
        result = frame
    elif group_col == "IncidentKey" and {"OrgId", "IncidentId"}.issubset(frame.columns):
        if frame["OrgId"].isna().any() or frame["IncidentId"].isna().any():
            raise ValueError("OrgId and IncidentId must be non-missing to build IncidentKey.")
        result = frame.copy()
        org = result["OrgId"].astype("string").str.strip()
        incident = result["IncidentId"].astype("string").str.strip()
        result[group_col] = org + "::" + incident
    else:
        required = "OrgId and IncidentId" if group_col == "IncidentKey" else group_col
        raise ValueError(f"Cannot build required grouping key '{group_col}'. Provide {required}.")
    if result[group_col].isna().any():
        raise ValueError(f"'{group_col}' contains missing values; cannot group incidents safely.")
    return result


def is_excluded_feature_column(column: str, group_col: str = GROUP_COL) -> bool:
    k = _key(column)
    return (
        k == _key(TARGET_COL)
        or k == _key(group_col)
        or k in LEAKAGE_COLUMNS
        or k in NON_FEATURE_ID_COLUMNS
    )


def infer_feature_types(
    frame: pd.DataFrame,
    group_col: str = GROUP_COL,
    target_col: str = TARGET_COL,
) -> tuple[list[str], list[str], list[str]]:
    """Return numeric features, categorical features, and excluded columns.

    Entirely missing columns and obvious date/time columns are excluded pending
    explicit feature engineering. This avoids learning arbitrary categories from
    raw timestamps; time features can be added deliberately in a later version.
    """
    excluded: list[str] = []
    candidates: list[str] = []

    for col in frame.columns:
        k = _key(col)
        if k == _key(target_col) or k == _key(group_col) or is_excluded_feature_column(col, group_col):
            excluded.append(col)
            continue
        if frame[col].isna().all():
            excluded.append(col)
            continue
        # A raw time/date string is high-cardinality and is not a meaningful
        # categorical feature by itself. Engineer derived time features only
        # after checking the dataset documentation and timestamp semantics.
        if any(token in k for token in ("timestamp", "datetime")) or k in {"date", "time", "eventdate", "eventtime"}:
            excluded.append(col)
            continue
        candidates.append(col)

    numeric: list[str] = []
    categorical: list[str] = []
    for col in candidates:
        k = _key(col)
        dtype = frame[col].dtype
        if k in FORCE_CATEGORICAL_COLUMNS:
            categorical.append(col)
        elif is_bool_dtype(dtype):
            categorical.append(col)
        elif is_numeric_dtype(dtype):
            numeric.append(col)
        else:
            categorical.append(col)

    return numeric, categorical, excluded


def validate_incident_labels(
    frame: pd.DataFrame,
    group_col: str = GROUP_COL,
    target_col: str = TARGET_COL,
) -> pd.Series:
    """Return one target per incident; fail if IDs/labels are missing or conflict."""
    frame = ensure_group_key(frame, group_col)
    missing = [c for c in (group_col, target_col) if c not in frame.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    if frame[group_col].isna().any():
        raise ValueError(f"'{group_col}' contains missing values; cannot group incidents safely.")
    if frame[target_col].isna().any():
        raise ValueError(f"'{target_col}' contains missing labels; resolve them before training.")

    unique_labels = frame.groupby(group_col, sort=False)[target_col].nunique(dropna=False)
    inconsistent = unique_labels[unique_labels > 1]
    if not inconsistent.empty:
        examples = inconsistent.index.astype(str).tolist()[:10]
        raise ValueError(
            f"{len(inconsistent)} incidents have conflicting '{target_col}' labels. "
            f"Resolve these incidents before training. Examples: {examples}"
        )
    return frame.groupby(group_col, sort=False)[target_col].first().rename(target_col)


def _stable_mode(series: pd.Series):
    values = series.dropna().astype("string")
    if values.empty:
        return pd.NA
    counts = values.value_counts(dropna=True)
    maximum = counts.max()
    # Deterministic tie-breaking, independent of source row order.
    winners = sorted(str(value) for value in counts[counts == maximum].index.tolist())
    return winners[0]


def normalize_feature_matrix(
    frame: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """Coerce incoming values to types consistent with the trained pipeline."""
    result = frame.loc[:, list(numeric_features) + list(categorical_features)].copy()
    for col in numeric_features:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype(np.float32)
    for col in categorical_features:
        # Convert categorical codes and strings into one stable representation.
        result[col] = result[col].astype("string")
    return result


def aggregate_incident_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    group_col: str = GROUP_COL,
) -> pd.DataFrame:
    """Build exactly one model input row per incident using the shared policy."""
    frame = ensure_group_key(frame, group_col)
    if group_col not in frame.columns:
        raise ValueError(f"Required incident identifier '{group_col}' is missing.")
    if frame[group_col].isna().any():
        raise ValueError(f"'{group_col}' contains missing values.")

    feature_columns = list(feature_columns)
    missing = [col for col in feature_columns if col not in frame.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required model features: " + ", ".join(missing[:30])
            + (" ..." if len(missing) > 30 else "")
        )

    num_cols = list(numeric_features)
    cat_cols = list(categorical_features)
    expected = set(num_cols) | set(cat_cols)
    if set(feature_columns) != expected:
        raise ValueError("Metadata feature schema is inconsistent: feature_columns must equal numeric + categorical features.")

    working = frame[[group_col] + feature_columns].copy()
    for col in num_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce").astype(np.float32)
    for col in cat_cols:
        working[col] = working[col].astype("string")

    aggregations = {col: "median" for col in num_cols}
    aggregations.update({col: _stable_mode for col in cat_cols})

    if aggregations:
        incident_features = (
            working.groupby(group_col, sort=False, dropna=False)
            .agg(aggregations)
            .reset_index()
        )
    else:
        incident_features = working[[group_col]].drop_duplicates().reset_index(drop=True)

    return normalize_feature_matrix(incident_features, num_cols, cat_cols).assign(
        **{group_col: incident_features[group_col].to_numpy()}
    )[[group_col] + feature_columns]


def build_incident_training_table(
    frame: pd.DataFrame,
    group_col: str = GROUP_COL,
    target_col: str = TARGET_COL,
    max_categories: int = 20,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Audit and aggregate raw GUIDE rows into one row per incident.

    High-cardinality categorical fields are retained but encoded with an
    encoder-level max_categories cap. Raw IDs and post-decision fields are
    excluded from model predictors. The returned table contains group_col,
    target_col, and the selected feature columns.
    """
    frame = ensure_group_key(frame, group_col)
    if group_col not in frame.columns or target_col not in frame.columns:
        raise ValueError(f"Dataset must include '{group_col}' and '{target_col}'.")

    labels = validate_incident_labels(frame, group_col, target_col)
    numeric, categorical, excluded = infer_feature_types(frame, group_col, target_col)
    if not numeric and not categorical:
        raise ValueError("No usable features remain after the feature audit.")

    # Report high-cardinality fields; the pipeline encoder caps output categories
    # per feature instead of dropping legitimate telemetry solely on cardinality.
    for col in categorical:
        unique_count = frame[col].nunique(dropna=False)
        if unique_count > max_categories:
            print(
                f"[!] '{col}' has {unique_count} raw categories; "
                f"OneHotEncoder will cap output at {max_categories} categories for this feature."
            )

    feature_columns = numeric + categorical
    incident_features = aggregate_incident_features(
        frame, feature_columns, numeric, categorical, group_col
    )
    target_frame = labels.reset_index()
    incident_table = incident_features.merge(
        target_frame, on=group_col, how="inner", validate="one_to_one", sort=False
    )

    if incident_table[group_col].duplicated().any():
        raise AssertionError("Incident aggregation did not produce a unique row per incident.")
    if incident_table.empty:
        raise ValueError("No incidents remained after aggregation.")

    return incident_table, numeric, categorical, excluded
