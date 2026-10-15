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
    """Normalize numeric and categorical values for scikit-learn."""
    result = frame.loc[
        :, list(numeric_features) + list(categorical_features)
    ].copy()

    for col in numeric_features:
        result[col] = pd.to_numeric(
            result[col], errors="coerce"
        ).astype(np.float32)

    for col in categorical_features:
        series = result[col].astype("string")
        result[col] = pd.Series(
            series.to_numpy(dtype=object, na_value=np.nan),
            index=result.index,
            dtype=object,
        )

    return result



def aggregate_incident_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    group_col: str = "IncidentKey",
) -> pd.DataFrame:
    """Aggregate source rows and calculate incident-level Aegis features."""

    frame = ensure_group_key(frame, group_col)

    if group_col not in frame.columns or frame[group_col].isna().any():
        raise ValueError(f"Invalid or missing incident key: {group_col}")

    feature_columns = list(feature_columns)
    num_cols = list(numeric_features)
    cat_cols = list(categorical_features)

    engineered = [
        "AegisSourceRowCount",
        "AegisUniqueAlertCount",
        "AegisUniqueDetectorCount",
    ]

    # Engineered columns are calculated from the raw source data.
    raw_features = [
        col for col in feature_columns if col not in engineered
    ]

    missing = [col for col in raw_features if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing raw features: {missing[:20]}")

    if set(feature_columns) != set(num_cols) | set(cat_cols):
        raise ValueError("Feature schema mismatch.")

    working = frame[[group_col] + raw_features].copy()

    for col in num_cols:
        if col not in engineered:
            working[col] = pd.to_numeric(
                working[col], errors="coerce"
            ).astype("float32")

    for col in cat_cols:
        working[col] = working[col].astype("string")

    aggregations = {
        col: "median" for col in num_cols if col not in engineered
    }
    aggregations.update({
        col: _stable_mode for col in cat_cols
    })

    if aggregations:
        result = (
            working.groupby(group_col, sort=False, dropna=False)
            .agg(aggregations)
            .reset_index()
        )
    else:
        result = (
            working[[group_col]].drop_duplicates().reset_index(drop=True)
        )

    grouped = frame.groupby(group_col, sort=False, dropna=False)
    source_keys = result[group_col]

    if "AegisSourceRowCount" in feature_columns:
        counts = grouped.size()
        result["AegisSourceRowCount"] = (
            source_keys.map(counts).astype("float32")
        )

    if "AegisUniqueAlertCount" in feature_columns:
        alert_col = next(
            (c for c in frame.columns if _key(c) == "alertid"), None
        )
        if alert_col is None:
            raise ValueError("AlertId is required for the alert-count feature.")

        counts = grouped[alert_col].nunique(dropna=True)
        result["AegisUniqueAlertCount"] = (
            source_keys.map(counts).astype("float32")
        )

    if "AegisUniqueDetectorCount" in feature_columns:
        detector_col = next(
            (c for c in frame.columns if _key(c) == "detectorid"), None
        )
        if detector_col is None:
            raise ValueError(
                "DetectorId is required for the detector-count feature."
            )

        counts = grouped[detector_col].nunique(dropna=True)
        result["AegisUniqueDetectorCount"] = (
            source_keys.map(counts).astype("float32")
        )

    incident_keys = result[group_col].to_numpy(copy=True)
    result = normalize_feature_matrix(result, num_cols, cat_cols)
    result[group_col] = incident_keys

    return result[[group_col] + feature_columns]



def build_incident_training_table(
    frame,
    group_col=GROUP_COL,
    target_col=TARGET_COL,
    max_categories=20,
):
    """Create one feature row per incident, including Aegis count features."""

    frame = ensure_group_key(frame, group_col)

    if group_col not in frame.columns or target_col not in frame.columns:
        raise ValueError(
            f"Dataset must include {group_col!r} and {target_col!r}."
        )

    labels = validate_incident_labels(frame, group_col, target_col)

    numeric, categorical, excluded = infer_feature_types(
        frame, group_col, target_col
    )

    engineered = [
        "AegisSourceRowCount",
        "AegisUniqueAlertCount",
        "AegisUniqueDetectorCount",
    ]

    # Add the generated counts as numeric predictors.
    numeric = [col for col in numeric if col not in engineered] + engineered
    feature_columns = numeric + categorical

    for col in categorical:
        unique_count = frame[col].nunique(dropna=False)
        if unique_count > max_categories:
            print(
                f"[!] {col!r} has {unique_count} categories; "
                f"the encoder will cap output at {max_categories}."
            )

    incident_features = aggregate_incident_features(
        frame,
        feature_columns,
        numeric,
        categorical,
        group_col,
    )

    incident_table = incident_features.merge(
        labels.reset_index(),
        on=group_col,
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if incident_table.empty:
        raise ValueError("No incidents remained after aggregation.")

    if incident_table[group_col].duplicated().any():
        raise AssertionError("Duplicate incident keys detected.")

    return incident_table, numeric, categorical, excluded
