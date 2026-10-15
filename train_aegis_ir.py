"""Train and evaluate ORION TRIAGE on Microsoft GUIDE-derived CSV data.

Prediction unit: one composite OrgId + IncidentId incident key. Multiple source rows are
aggregated with the same helper used by the Streamlit app. No synthetic fallback is provided.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aegis_features import (
    GROUP_COL,
    TARGET_COL,
    build_incident_training_table,
    ensure_group_key,
    validate_incident_labels,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "plots"
DEFAULT_DATASET = ROOT / "guide_sample.csv"
MAX_CATEGORIES = 20
EXPECTED_CLASSES = {"TruePositive", "BenignPositive", "FalsePositive"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to a GUIDE-derived CSV")
    parser.add_argument("--max-categories", type=int, default=MAX_CATEGORIES)
    return parser.parse_args()


def build_pipeline(numeric_features, categorical_features, model):
    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                max_categories=MAX_CATEGORIES,
                dtype=np.float32,
            )),
        ])
        transformers.append(("cat", categorical_transformer, categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def validate_folds(cv_splits, y, label):
    all_classes = set(y.astype(str).unique())
    for fold_number, (fit_idx, held_idx) in enumerate(cv_splits, start=1):
        fit_classes = set(y.iloc[fit_idx].astype(str).unique())
        held_classes = set(y.iloc[held_idx].astype(str).unique())
        if fit_classes != all_classes:
            raise ValueError(f"{label} fold {fold_number}: training partition is missing a class.")
        if held_classes != all_classes:
            raise ValueError(f"{label} fold {fold_number}: validation partition is missing a class.")


def save_plots(incident_table, numeric_features, y_test, y_pred, y_proba,
               class_order, cv_df, winner, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    class_counts = incident_table[TARGET_COL].value_counts().reindex(class_order, fill_value=0)
    class_counts.plot(kind="bar", ax=ax)
    ax.set_title("ORION TRIAGE — Incident Grade Distribution")
    ax.set_xlabel("Incident Grade")
    ax.set_ylabel("Number of Incidents")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_class_distribution.png", dpi=220)
    plt.close(fig)

    usable_numeric = [c for c in numeric_features if c in incident_table.columns]
    # Limit the plotted subset to keep the chart readable.
    heatmap_features = usable_numeric[:20]
    fig, ax = plt.subplots(figsize=(max(8, len(heatmap_features) * 0.55), 6))
    if heatmap_features:
        corr = incident_table[heatmap_features].corr()
        sns.heatmap(corr, annot=len(heatmap_features) <= 12, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Audited Numeric Feature Correlations (First 20 Features)")
    else:
        ax.text(0.5, 0.5, "No numeric features available", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(plot_dir / "02_correlation_heatmap.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred, labels=class_order)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_order, yticklabels=class_order, ax=ax)
    ax.set_title(f"Held-Out Incident Confusion Matrix — {winner}")
    ax.set_xlabel("Predicted Verdict")
    ax.set_ylabel("Actual Verdict")
    fig.tight_layout()
    fig.savefig(plot_dir / "03_confusion_matrix.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_test_values = y_test.to_numpy()
    for index, class_name in enumerate(class_order):
        one_vs_rest = (y_test_values == class_name).astype(int)
        RocCurveDisplay.from_predictions(
            one_vs_rest, y_proba[:, index], name=f"{class_name} (OvR)", ax=ax
        )
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_title("Held-Out Multiclass One-vs-Rest ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.tight_layout()
    fig.savefig(plot_dir / "04_roc_curves.png", dpi=220)
    plt.close(fig)

    cv_plot = cv_df.reset_index(names="Model").melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=cv_plot, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Group-Aware 5-Fold Cross-Validation Comparison")
    ax.set_ylabel("Mean CV Score")
    fig.tight_layout()
    fig.savefig(plot_dir / "05_model_comparison.png", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\nDownload the GUIDE data and place your prepared CSV here, "
            "or pass --dataset /path/to/your.csv. No synthetic fallback is used."
        )

    if args.max_categories < 2:
        raise ValueError("--max-categories must be at least 2.")

    global MAX_CATEGORIES
    MAX_CATEGORIES = args.max_categories

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Loading: {dataset_path}")
    raw_df = pd.read_csv(dataset_path, low_memory=False)
    raw_df = ensure_group_key(raw_df, GROUP_COL)
    print(f"[+] Raw rows: {len(raw_df):,}")

    # Audit raw incident label consistency before building the incident table.
    label_by_incident = validate_incident_labels(raw_df)
    rows_per_incident = raw_df.groupby(GROUP_COL, sort=False).size()
    print(f"[+] Unique incidents: {raw_df[GROUP_COL].nunique():,}")
    print(f"[+] Rows per incident — median: {rows_per_incident.median():.2f}; max: {rows_per_incident.max()}")
    if len(raw_df) != len(label_by_incident):
        print("[+] Multiple source rows detected; applying numeric-median / categorical-mode aggregation.")

    incident_table, numeric_features, categorical_features, excluded_columns = build_incident_training_table(
        raw_df, max_categories=MAX_CATEGORIES
    )
    class_counts = incident_table[TARGET_COL].value_counts()
    found_classes = set(incident_table[TARGET_COL].astype(str).unique())
    if found_classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected all GUIDE classes {sorted(EXPECTED_CLASSES)}, found {sorted(found_classes)}. "
            "Check the dataset or sample construction."
        )
    if class_counts.min() < 5:
        raise ValueError("At least five incidents per class are required for 5-fold grouped evaluation.")

    print(f"[+] Incident-level dataset: {len(incident_table):,} incidents; {len(numeric_features)} numeric + {len(categorical_features)} categorical features")
    encoded_column_cap = len(numeric_features) + len(categorical_features) * MAX_CATEGORIES
    estimated_dense_mb = len(incident_table) * encoded_column_cap * np.dtype(np.float32).itemsize / (1024 ** 2)
    print(f"[~] Estimated dense transformed-matrix size: ~{estimated_dense_mb:.1f} MiB "
          f"(matrix estimate only; excludes copies and model/training overhead)")
    if estimated_dense_mb > 1500:
        print("[!] Large dense matrix estimate. Reduce the sample size or use a sparse-compatible model pipeline before training.")
    print(f"[-] Excluded fields: {excluded_columns}")
    print("[+] Incident class proportions:")
    print(incident_table[TARGET_COL].value_counts(normalize=True).round(4).to_dict())

    feature_columns = numeric_features + categorical_features
    X = incident_table[feature_columns].copy()
    y = incident_table[TARGET_COL].astype(str).copy()
    groups = incident_table[GROUP_COL].copy()

    # A five-fold stratified group split yields an approximately 80/20 holdout.
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_splits = list(outer_cv.split(X, y, groups))
    train_idx, test_idx = outer_splits[0]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
    if not set(groups_train).isdisjoint(set(groups_test)):
        raise AssertionError("Incident ID overlap detected between training and test partitions.")
    print(f"[+] Holdout: {len(X_train):,} train incidents / {len(X_test):,} test incidents")
    print("[+] Training proportions:", y_train.value_counts(normalize=True).round(3).to_dict())
    print("[+] Test proportions:", y_test.value_counts(normalize=True).round(3).to_dict())

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=73)
    cv_splits = list(sgkf.split(X_train, y_train, groups_train))
    validate_folds(cv_splits, y_train, "Model-selection CV")

    class_scorers = {
        "accuracy": "accuracy",
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    }
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
    }

    cv_results = {}
    fitted_pipelines = {}
    for name, estimator in models.items():
        print(f"[+] CV: {name}")
        candidate = build_pipeline(numeric_features, categorical_features, estimator)
        results = cross_validate(
            candidate, X_train, y_train,
            groups=groups_train, cv=cv_splits, scoring=class_scorers,
            return_train_score=False, error_score="raise", n_jobs=1,
        )
        cv_results[name] = {
            "Accuracy": float(np.mean(results["test_accuracy"])),
            "Macro F1": float(np.mean(results["test_f1_macro"])),
            "Macro Precision": float(np.mean(results["test_precision_macro"])),
            "Macro Recall": float(np.mean(results["test_recall_macro"])),
            "Macro F1 Std": float(np.std(results["test_f1_macro"], ddof=1)),
        }
        fitted_pipelines[name] = candidate

    cv_df = pd.DataFrame(cv_results).T
    print("\n--- GROUP-AWARE CV RESULTS ---")
    print(cv_df.round(4))
    winner = cv_df["Macro F1"].idxmax()
    print(f"[+] Selected by mean CV Macro F1: {winner}")

    calibration_splits = list(sgkf.split(X_train, y_train, groups_train))
    validate_folds(calibration_splits, y_train, "Calibration CV")
    calibrated_pipeline = CalibratedClassifierCV(
        estimator=build_pipeline(numeric_features, categorical_features, models[winner]),
        cv=calibration_splits,
        method="sigmoid",
        ensemble=False,
    )
    calibrated_pipeline.fit(X_train, y_train)
    class_order = [str(c) for c in calibrated_pipeline.classes_]
    if set(y_test.unique()) != set(class_order):
        raise ValueError("Held-out test set does not contain every class; inspect split/class support.")

    y_pred = calibrated_pipeline.predict(X_test)
    y_proba = calibrated_pipeline.predict_proba(X_test)
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc_ovr_macro": float(roc_auc_score(y_test, y_proba, labels=class_order, multi_class="ovr", average="macro")),
        "test_incidents": int(len(X_test)),
    }
    print("\n--- HELD-OUT INCIDENT-LEVEL TEST RESULTS ---")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

    # Keep the illustrative abstention threshold separate from model selection.
    confidence = np.max(y_proba, axis=1)
    threshold_analysis = {}
    for threshold in (0.50, 0.60, 0.65, 0.70, 0.80, 0.90):
        accepted = confidence >= threshold
        threshold_analysis[str(threshold)] = {
            "coverage": float(accepted.mean()),
            "review_rate": float((~accepted).mean()),
            "accepted_accuracy": float(accuracy_score(y_test[accepted], y_pred[accepted])) if accepted.any() else None,
            "accepted_count": int(accepted.sum()),
        }

    model_path = MODEL_DIR / "aegis_ir_pipeline.joblib"
    joblib.dump(calibrated_pipeline, model_path)
    metadata = {
        "project": "ORION TRIAGE",
        "model_version": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "winning_model_name": winner,
        "target_col": TARGET_COL,
        "group_col": GROUP_COL,
        "class_order": class_order,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": feature_columns,
        "excluded_columns": excluded_columns,
        "aggregation_policy": {
            "prediction_unit": "incident",
            "numeric": "median across source rows per incident",
            "categorical": "deterministic mode across source rows per incident",
        },
        "max_categories_per_categorical_feature": MAX_CATEGORIES,
        "cv_benchmarks": {
            name: {metric: float(value) for metric, value in values.items()}
            for name, values in cv_results.items()
        },
        "test_metrics": test_metrics,
        "threshold_analysis": threshold_analysis,
        "source_dataset_name": dataset_path.name,
    }
    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Export raw source rows for held-out incidents; app applies the same aggregation policy.
    held_out_ids = set(groups_test.astype(str))
    raw_group_as_str = raw_df[GROUP_COL].astype(str)
    demo = raw_df.loc[raw_group_as_str.isin(held_out_ids)].copy()
    remove_from_demo = [c for c in [TARGET_COL] if c in demo.columns]
    remove_from_demo += [c for c in demo.columns if c.lower().replace("_", "") in {
        "lastverdict", "actiongrouped", "actiongranular", "remediationstate", "resolutiontime"
    }]
    demo = demo.drop(columns=list(dict.fromkeys(remove_from_demo)), errors="ignore")
    demo_path = ROOT / "demo_queue.csv"
    demo.to_csv(demo_path, index=False)

    save_plots(incident_table, numeric_features, y_test, y_pred, y_proba, class_order, cv_df, winner, PLOT_DIR)
    print(f"[+] Saved model: {model_path}")
    print(f"[+] Saved metadata: {metadata_path}")
    print(f"[+] Saved held-out demo queue: {demo_path}")
    print(f"[+] Saved five plots in: {PLOT_DIR}")
    print("[!] Metrics are from the supplied dataset and have not been independently validated against a production SOC.")


if __name__ == "__main__":
    main()
