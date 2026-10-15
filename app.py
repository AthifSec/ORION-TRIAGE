"""ORION TRIAGE — human-supervised SOC incident triage workbench."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from aegis_app_logic import initialize_feedback_db, read_feedback, save_feedback, score_input
from aegis_catboost_logic import score_input_catboost

@st.cache_data(show_spinner="Scoring incidents...")
def cached_score_input(
    raw_df: pd.DataFrame,
    _pipeline: Any,
    metadata: dict,
    model_version: str,
):
    """Reuse incident predictions until the input or model version changes."""
    if metadata.get("model_family") == "catboost_row_level":
        return score_input_catboost(raw_df, _pipeline, metadata)
    return score_input(raw_df, _pipeline, metadata)



ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "aegis_ir_pipeline.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
CATBOOST_MODEL_PATH = ROOT / "models" / "aegis_ir_candidate_g.cbm"
CATBOOST_META_PATH = ROOT / "models" / "catboost_candidate_g_metadata.json"
DEMO_PATH = ROOT / "demo_queue.csv"
DB_PATH = ROOT / "analyst_feedback.db"

st.set_page_config(page_title="ORION TRIAGE | Analyst Workbench", page_icon="🛡️", layout="wide")

ROUTE_REVIEW = "Needs Analyst Review"
ROUTE_SUGGESTION = "High-Confidence Suggestion — Approval Required"


@st.cache_resource
def load_model(path_string: str, modified_time: float):
    # The modification time participates in the cache key so retraining refreshes the artifact.
    del modified_time
    path = Path(path_string)
    # joblib files are pickle-based: only load artifacts created by a trusted local training run.
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_catboost_model(path_string: str, modified_time: float):
    """Load the optional Candidate G CatBoost model."""
    del modified_time
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(path_string)
    return model


@st.cache_data
def load_metadata(path_string: str, modified_time: float):
    del modified_time
    path = Path(path_string)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_demo_csv(path_string: str, modified_time: float):
    del modified_time  # Included in cache key so replacing the demo file refreshes this data.
    return pd.read_csv(path_string, low_memory=False)


def local_shap_explanation(pipeline, reference_features: pd.DataFrame, row: pd.DataFrame,
                           feature_cols: list[str], numeric_features: list[str],
                           categorical_features: list[str], class_index: int) -> pd.DataFrame:
    """Compute optional model-agnostic SHAP values for one row.

    SHAP's permutation explainer operates on a numeric explanation matrix. To
    support mixed string/numeric telemetry without pretending category codes
    are ordered, categorical values are encoded only for masking and decoded
    back to their original values before every model call. The estimator still
    receives the original semantic feature types.
    """
    import shap  # Optional explanation dependency; import only when requested.

    background_frame = reference_features[feature_cols].copy()
    background_frame = background_frame.sample(n=min(8, len(background_frame)), random_state=42)
    row_frame = row[feature_cols].copy()

    # Establish a reversible codebook for every categorical column, based on all
    # available queue values plus the selected row. Missingness gets its own code.
    missing_token = "__AEGIS_MISSING__"
    category_maps: dict[str, dict[str, int]] = {}
    reverse_maps: dict[str, dict[int, object]] = {}

    def token(value):
        return missing_token if pd.isna(value) else str(value)

    for col in categorical_features:
        values = pd.concat([reference_features[col], row[col]], ignore_index=True)
        tokens = sorted({token(v) for v in values.tolist()})
        category_maps[col] = {value: index for index, value in enumerate(tokens)}
        reverse_maps[col] = {index: (np.nan if value == missing_token else value) for value, index in category_maps[col].items()}

    def encode(frame):
        result = pd.DataFrame(index=frame.index)
        for col in numeric_features:
            numeric = pd.to_numeric(frame[col], errors="coerce")
            fallback = pd.to_numeric(reference_features[col], errors="coerce").median()
            if pd.isna(fallback):
                fallback = 0.0
            result[col] = numeric.fillna(fallback).astype(float)
        for col in categorical_features:
            result[col] = frame[col].map(lambda value: category_maps[col][token(value)]).astype(float)
        return result[feature_cols].to_numpy(dtype=float)

    def decode_and_predict(values):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        decoded = pd.DataFrame(values, columns=feature_cols)
        for col in numeric_features:
            decoded[col] = pd.to_numeric(decoded[col], errors="coerce").astype(np.float32)
        for col in categorical_features:
            reverse = reverse_maps[col]
            codes = np.rint(decoded[col].to_numpy()).astype(int)
            decoded[col] = [reverse.get(int(code), np.nan) for code in codes]
            categorical_values = decoded[col].astype("string")
            decoded[col] = pd.Series(
                categorical_values.to_numpy(dtype=object, na_value=np.nan),
                index=decoded.index,
                dtype=object,
            )
        return pipeline.predict_proba(decoded[feature_cols])

    background_values = encode(background_frame)
    row_values = encode(row_frame)
    explainer = shap.Explainer(
        decode_and_predict,
        background_values,
        algorithm="permutation",
        feature_names=feature_cols,
    )
    max_evals = 2 * len(feature_cols) + 1
    explanation = explainer(row_values, max_evals=max_evals)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        contribution = values[0, :, class_index]
    elif values.ndim == 2:
        contribution = values[0, :]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
    return pd.DataFrame({"Feature": feature_cols, "SHAP contribution": contribution}).assign(
        Absolute=lambda table: table["SHAP contribution"].abs()
    ).sort_values("Absolute", ascending=False)


initialize_feedback_db(DB_PATH)

# Candidate G is preferred when its optional model artifact and metadata are
# present and load successfully. The original Random Forest remains a fallback.
model_load_warning = None
pipeline = None
metadata = None

if CATBOOST_MODEL_PATH.exists() and CATBOOST_META_PATH.exists():
    try:
        pipeline = load_catboost_model(
            str(CATBOOST_MODEL_PATH), CATBOOST_MODEL_PATH.stat().st_mtime
        )
        metadata = load_metadata(
            str(CATBOOST_META_PATH), CATBOOST_META_PATH.stat().st_mtime
        )
        if not metadata or metadata.get("model_family") != "catboost_row_level":
            raise ValueError("Candidate G metadata has the wrong model_family.")
    except Exception as exc:
        model_load_warning = (
            f"Candidate G could not be loaded ({exc}). Falling back to Random Forest."
        )
        pipeline = None
        metadata = None

if pipeline is None or metadata is None:
    pipeline = load_model(
        str(MODEL_PATH), MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
    )
    metadata = load_metadata(
        str(META_PATH), META_PATH.stat().st_mtime if META_PATH.exists() else 0.0
    )

st.title("🛡️ ORION TRIAGE")
st.caption("Evidence-grounded, human-supervised incident triage decision support")

if model_load_warning:
    st.warning(model_load_warning)

if pipeline is None or metadata is None:
    st.error(
        "No compatible model artifacts were found. Restore the Random Forest artifacts "
        "or install CatBoost and place Candidate G model and metadata in the documented paths."
    )
    st.stop()

st.caption(f"Active model: {metadata.get('winning_model_name', 'unknown')} · Version: {metadata.get('model_version', 'unknown')}")

with st.sidebar:
    st.header("Control Panel")
    source_mode = st.radio("Telemetry source", ["Upload CSV", "Held-out demo queue"])
    uploaded_file = None
    if source_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload incoming telemetry", type=["csv"])
    threshold = st.slider(
        "Uncertainty review threshold",
        min_value=0.50, max_value=0.99, value=0.65, step=0.01,
        help="Predictions below this maximum class probability are sent to analyst review. This default is illustrative, not a validated safety threshold.",
    )

if source_mode == "Held-out demo queue":
    if not DEMO_PATH.exists():
        st.warning("No demo_queue.csv found. Run the training script to export source rows from its held-out partition, or choose Upload CSV.")
        st.stop()
    raw_queue = load_demo_csv(str(DEMO_PATH), DEMO_PATH.stat().st_mtime)
    source_name = DEMO_PATH.name
elif uploaded_file is not None:
    try:
        raw_queue = pd.read_csv(uploaded_file, low_memory=False)
        source_name = uploaded_file.name
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()
else:
    st.info("Upload a CSV to start triage.")
    st.stop()

try:
    queue_df, group_col = cached_score_input(
        raw_queue, pipeline, metadata,
        str(metadata.get("model_version", "unknown")),
    )
except Exception as exc:
    st.error(f"Input validation or scoring failed: {exc}")
    st.stop()

queue_df["Routing"] = np.where(
    queue_df["Model_Confidence"] >= threshold,
    ROUTE_SUGGESTION,
    ROUTE_REVIEW,
)

with st.sidebar:
    selected_routes = st.multiselect(
        "Queue status filter",
        options=[ROUTE_REVIEW, ROUTE_SUGGESTION],
        default=[ROUTE_REVIEW, ROUTE_SUGGESTION],
    )

st.caption(
    f"Model version: {metadata.get('model_version', 'unknown')} · "
    f"Prediction unit: {metadata.get('aggregation_policy', {}).get('prediction_unit', 'incident')} · "
    f"Source: {source_name}"
)

queue_tab, health_tab, audit_tab = st.tabs(["Incident Queue", "Model Health", "Feedback Audit"])

with queue_tab:
    filtered = queue_df[queue_df["Routing"].isin(selected_routes)].copy()
    total, review, suggestions = st.columns(3)
    total.metric("Incidents scored", f"{len(queue_df):,}")
    review.metric("Needs analyst review", f"{int((queue_df['Routing'] == ROUTE_REVIEW).sum()):,}")
    suggestions.metric("High-confidence suggestions", f"{int((queue_df['Routing'] == ROUTE_SUGGESTION).sum()):,}")

    identity_cols = [col for col in ("OrgId", "IncidentId", group_col) if col in filtered.columns]
    identity_cols = list(dict.fromkeys(identity_cols))
    queue_columns = identity_cols + ["Model_Prediction", "Model_Confidence", "Routing"]
    queue_view = filtered[queue_columns].rename(columns={
        "OrgId": "Organization ID",
        "IncidentId": "Source Incident ID",
        group_col: "Incident Key",
        "Model_Prediction": "Suggested verdict",
        "Model_Confidence": "Top-class probability",
    })
    st.dataframe(
        queue_view.style.background_gradient(subset=["Top-class probability"], cmap="RdYlGn", vmin=0, vmax=1),
        use_container_width=True,
        hide_index=True,
    )

    if filtered.empty:
        st.info("No incidents match the current routing filters.")
    else:
        selected_incident = st.selectbox("Investigate incident", filtered[group_col].astype(str).tolist())
        record = filtered.loc[filtered[group_col].astype(str) == selected_incident].iloc[0]
        left, right = st.columns([1.4, 1])

        with left:
            st.subheader(f"Incident {selected_incident}")
            feature_cols = metadata["feature_columns"]
            if metadata.get("model_family") == "catboost_row_level":
                st.markdown(
                    "**Incident-level summary** — categorical values below are deterministic "
                    "modes across all source rows; Candidate G scores up to four sampled source rows."
                )
            else:
                st.markdown("**Aggregated model inputs**")
            st.json({key: (None if pd.isna(record[key]) else record[key].item() if isinstance(record[key], np.generic) else record[key]) for key in feature_cols if key in record})

            probability_dict = {cls: float(record[f"Prob_{cls}"]) for cls in metadata["class_order"]}
            fig, ax = plt.subplots(figsize=(7, 3))
            sns.barplot(x=list(probability_dict.values()), y=list(probability_dict.keys()), ax=ax)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Model class probability")
            ax.set_ylabel("")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            incident_hash = hashlib.sha1(str(selected_incident).encode("utf-8")).hexdigest()[:12]
            signature_payload = json.dumps({
                key: (None if pd.isna(record[key]) else str(record[key]))
                for key in metadata["feature_columns"] if key in record
            }, sort_keys=True)
            record_signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()[:10]
            shap_state_key = f"shap_result_{incident_hash}_{record_signature}"
            if metadata.get("model_family") == "catboost_row_level":
                st.info(
                    "Local SHAP is not enabled for Candidate G yet: the current explanation "
                    "helper is designed for the aggregated Random Forest input, not the "
                    "row-level probability-averaging model."
                )
            if metadata.get("model_family") != "catboost_row_level" and st.button("Calculate local SHAP explanation", key=f"shap_button_{incident_hash}_{record_signature}"):
                with st.spinner("Computing a model-agnostic explanation; this may take a moment..."):
                    try:
                        st.session_state[shap_state_key] = local_shap_explanation(
                            pipeline,
                            queue_df[metadata["feature_columns"]],
                            record.to_frame().T,
                            metadata["feature_columns"],
                            metadata["numeric_features"],
                            metadata["categorical_features"],
                            metadata["class_order"].index(str(record["Model_Prediction"])),
                        )
                    except ImportError:
                        st.warning("SHAP is not installed. Install requirements.txt to enable explanations.")
                    except Exception as exc:
                        st.warning(f"SHAP explanation could not be computed for this row: {exc}")
            if metadata.get("model_family") != "catboost_row_level" and shap_state_key in st.session_state:
                st.caption("Positive values push the explained class probability upward relative to the explainer's background; they are not causal effects.")
                top = st.session_state[shap_state_key].head(15).sort_values("SHAP contribution")
                fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(top))))
                ax.barh(top["Feature"], top["SHAP contribution"])
                ax.axvline(0, linewidth=1, color="black")
                ax.set_xlabel("SHAP contribution to predicted-class probability")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with right:
            st.subheader("Human review")
            st.info(
                f"**Model suggestion:** {record['Model_Prediction']}\n\n"
                f"**Top-class probability:** {record['Model_Confidence']:.1%}\n\n"
                f"**Routing:** {record['Routing']}\n\n"
                "No action is taken automatically."
            )
            with st.form("analyst_review_form", clear_on_submit=True):
                verdict_index = metadata["class_order"].index(str(record["Model_Prediction"]))
                verdict = st.selectbox("Analyst-adjudicated verdict", metadata["class_order"], index=verdict_index)
                notes = st.text_area("Investigation notes", placeholder="Record corroborating evidence, approved activity, or reasons for disagreement.")
                acknowledged = st.checkbox("I reviewed the available evidence and understand this is a model suggestion.")
                submitted = st.form_submit_button("Save analyst decision", type="primary")
                if submitted:
                    if not acknowledged:
                        st.error("Please acknowledge the review before saving the decision.")
                    else:
                        probabilities = {cls: float(record[f"Prob_{cls}"]) for cls in metadata["class_order"]}
                        save_feedback(
                            DB_PATH,
                            incident_id=str(selected_incident),
                            model_version=str(metadata.get("model_version", "unknown")),
                            prediction=str(record["Model_Prediction"]),
                            confidence=float(record["Model_Confidence"]),
                            probabilities=probabilities,
                            verdict=str(verdict),
                            notes=str(notes),
                            source_name=source_name,
                        )
                        st.success("Analyst decision saved to the local audit database. It has not retrained the model.")

with health_tab:
    st.subheader("Offline validation results")
    st.caption("These metrics describe the held-out incident-level partition recorded at training time. They are not live queue-performance metrics.")
    test_metrics = metadata.get("test_metrics", {})
    metric_columns = st.columns(5)
    for column, label, key in zip(
        metric_columns,
        ["Macro F1", "OvR Macro ROC-AUC", "Macro Precision", "Macro Recall", "Accuracy"],
        ["macro_f1", "roc_auc_ovr_macro", "macro_precision", "macro_recall", "accuracy"],
    ):
        value = test_metrics.get(key)
        column.metric(label, "N/A" if value is None else f"{value:.3f}")

    st.subheader("Cross-validation benchmarks")
    cv_benchmarks = metadata.get("cv_benchmarks", {})
    if cv_benchmarks:
        cv_table = pd.DataFrame(cv_benchmarks).T
        st.dataframe(cv_table, use_container_width=True)
    else:
        st.info("No CV benchmark table was saved in the metadata.")

    st.subheader("Review-threshold trade-off")
    threshold_analysis = metadata.get("threshold_analysis", {})
    if threshold_analysis:
        threshold_table = pd.DataFrame(threshold_analysis).T.reset_index(names="Threshold")
        threshold_table["Threshold"] = pd.to_numeric(threshold_table["Threshold"])
        st.dataframe(threshold_table, use_container_width=True, hide_index=True)
        st.caption("Threshold results are historical measurements on the held-out partition. Changing the sidebar threshold does not change these historical measurements.")

with audit_tab:
    st.subheader("Analyst feedback history")
    feedback = read_feedback(DB_PATH, limit=200)
    if feedback.empty:
        st.info("No analyst decisions have been logged yet.")
    else:
        st.dataframe(feedback, use_container_width=True, hide_index=True)
        csv_bytes = feedback.to_csv(index=False).encode("utf-8")
        st.download_button("Download visible audit history as CSV", data=csv_bytes, file_name="aegis_ir_feedback_audit.csv", mime="text/csv")

st.divider()
st.caption("Prototype disclaimer: decision support only. Do not connect this prototype to automatic containment or incident-closure actions. Review dataset licensing, privacy, access controls, and deployment security before operational use.")
