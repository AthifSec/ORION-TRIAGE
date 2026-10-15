"""Create a stratified, incident-complete sample from Microsoft's GUIDE_train.csv.

This script scans the large source CSV twice in chunks: first to map incident keys
and labels, then to retain every source row for a stratified sample of incidents.
It does not create synthetic records or change labels.

Example:
    python prepare_guide_sample.py --input GUIDE_train.csv --output guide_sample.csv --incidents 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

TARGET_COL = "IncidentGrade"
ORG_COL = "OrgId"
INCIDENT_COL = "IncidentId"
KEY_COL = "IncidentKey"
EXPECTED_CLASSES = {"TruePositive", "BenignPositive", "FalsePositive"}


def build_key(frame: pd.DataFrame) -> pd.Series:
    missing = [col for col in (ORG_COL, INCIDENT_COL) if col not in frame.columns]
    if missing:
        raise ValueError(f"GUIDE source is missing required key columns: {missing}")
    if frame[ORG_COL].isna().any() or frame[INCIDENT_COL].isna().any():
        raise ValueError("Cannot construct IncidentKey from missing OrgId/IncidentId values.")
    return frame[ORG_COL].astype("string").str.strip() + "::" + frame[INCIDENT_COL].astype("string").str.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to GUIDE_train.csv")
    parser.add_argument("--output", default="guide_sample.csv", help="Output sample CSV path")
    parser.add_argument("--incidents", type=int, default=20000, help="Number of unique incidents to sample")
    parser.add_argument("--chunksize", type=int, default=150000, help="CSV rows per chunk")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GUIDE source file not found: {source}")
    if args.incidents < 30:
        raise ValueError("Choose at least 30 incidents to support a meaningful 5-fold evaluation.")
    if args.chunksize < 1000:
        raise ValueError("--chunksize must be at least 1000.")

    print(f"[1/2] Scanning incident keys and labels from {source.name}")
    label_by_key: dict[str, str] = {}
    conflicting_keys: set[str] = set()
    usecols = [ORG_COL, INCIDENT_COL, TARGET_COL]
    dtypes = {ORG_COL: "string", INCIDENT_COL: "string", TARGET_COL: "string"}
    chunks_seen = 0

    for chunk in pd.read_csv(source, usecols=usecols, dtype=dtypes, chunksize=args.chunksize):
        chunks_seen += 1
        chunk = chunk.dropna(subset=usecols)
        if chunk.empty:
            continue
        chunk[KEY_COL] = build_key(chunk)
        pairs = chunk[[KEY_COL, TARGET_COL]].drop_duplicates()
        for key, label in pairs.itertuples(index=False, name=None):
            key = str(key)
            label = str(label)
            previous = label_by_key.get(key)
            if previous is None:
                label_by_key[key] = label
            elif previous != label:
                conflicting_keys.add(key)
        if chunks_seen % 10 == 0:
            print(f"  scanned {chunks_seen:,} chunks; found {len(label_by_key):,} incident keys")

    for key in conflicting_keys:
        label_by_key.pop(key, None)
    labels = pd.DataFrame({KEY_COL: list(label_by_key.keys()), TARGET_COL: list(label_by_key.values())})
    labels = labels[labels[TARGET_COL].isin(EXPECTED_CLASSES)].reset_index(drop=True)
    if labels.empty:
        raise ValueError("No usable incident labels found. Verify that this is GUIDE_train.csv.")
    actual_classes = set(labels[TARGET_COL].unique())
    if actual_classes != EXPECTED_CLASSES:
        raise ValueError(f"All three IncidentGrade labels are required; found {sorted(actual_classes)}")
    if conflicting_keys:
        print(f"[!] Excluding {len(conflicting_keys):,} incident keys with inconsistent labels.")

    sample_size = min(args.incidents, len(labels))
    if labels[TARGET_COL].value_counts().min() < 5:
        raise ValueError("At least five incidents per class are needed for five-fold validation.")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=args.random_state)
    selected_indices, _ = next(splitter.split(labels[[KEY_COL]], labels[TARGET_COL]))
    selected_keys = set(labels.iloc[selected_indices][KEY_COL].astype(str))
    print(f"[+] Selected {len(selected_keys):,} unique incidents with stratified class sampling.")

    print(f"[2/2] Extracting all source rows for selected incidents to {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    output_rows = 0
    written_header = False

    # Read all features in chunks while forcing the key/label types to stay stable.
    for chunk_number, chunk in enumerate(pd.read_csv(source, dtype=dtypes, low_memory=False, chunksize=args.chunksize), start=1):
        chunk = chunk.dropna(subset=[ORG_COL, INCIDENT_COL, TARGET_COL])
        if chunk.empty:
            continue
        chunk[KEY_COL] = build_key(chunk)
        selected = chunk[chunk[KEY_COL].isin(selected_keys)]
        if selected.empty:
            continue
        selected.to_csv(output, mode="a" if written_header else "w", index=False, header=not written_header)
        written_header = True
        output_rows += len(selected)
        if chunk_number % 10 == 0:
            print(f"  scanned {chunk_number:,} chunks; exported {output_rows:,} rows")

    if not written_header:
        raise RuntimeError("No records were exported; verify key and label columns in the source file.")

    # Lightweight output audit. Keep this read chunked for large outputs.
    sample_labels: dict[str, str] = {}
    sample_rows = 0
    for chunk in pd.read_csv(output, usecols=[KEY_COL, TARGET_COL], dtype={KEY_COL: "string", TARGET_COL: "string"}, chunksize=args.chunksize):
        sample_rows += len(chunk)
        for key, label in chunk[[KEY_COL, TARGET_COL]].drop_duplicates().itertuples(index=False, name=None):
            previous = sample_labels.get(str(key))
            if previous is not None and previous != str(label):
                raise ValueError(f"Sample output contains conflicting labels for {key}")
            sample_labels[str(key)] = str(label)
    counts = pd.Series(list(sample_labels.values())).value_counts().to_dict()
    print("\nSAMPLE READY")
    print(f"Output: {output}")
    print(f"Source rows retained: {sample_rows:,}")
    print(f"Unique incidents retained: {len(sample_labels):,}")
    print(f"Incident class counts: {counts}")
    print("No synthetic data was generated. The source labels were retained unchanged.")


if __name__ == "__main__":
    main()
