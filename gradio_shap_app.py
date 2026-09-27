"""Gradio app for loading the saved best model and explaining one sample.

Expected deployment artifact from the notebook:

    artifacts/best_model_deploy.joblib

Run:

    python gradio_shap_app.py

Optional environment variables:

    MODEL_ARTIFACT=/path/to/best_model.joblib
    SAMPLE_FILE=/path/to/model_ready_samples.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from semopy import Model
from sklearn.pipeline import Pipeline


APP_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_CANDIDATES = [
    APP_DIR / "artifacts" / "best_model_deploy.joblib",
    APP_DIR / "artifacts" / "best_model.joblib",
    APP_DIR / "best_model_deploy.joblib",
    APP_DIR / "best_model.joblib",
    APP_DIR / "artifacts" / "final_pipeline.joblib",
    APP_DIR / "final_pipeline.joblib",
]
DEFAULT_SAMPLE_CANDIDATES = [
    APP_DIR / "artifacts" / "model_ready_samples.parquet",
    APP_DIR / "artifacts" / "model_ready_samples.csv",
    APP_DIR / "artifacts" / "samples.parquet",
    APP_DIR / "artifacts" / "samples.csv",
    APP_DIR / "data_survey.xlsx",
]

SEM_NODE_LABELS = {
    "AcademicStress": "Academic\nStress",
    "FinancialStress": "Financial\nStress",
    "SocialIsolation": "Social\nIsolation",
    "VictimizationTrauma": "Victimization\nTrauma",
    "Anxiety": "Anxiety",
    "Depression": "Predicted\nPHQ",
    "ObservedCovariates": "Observed\nCovariates",
}

SEM_NODE_POSITIONS = {
    "AcademicStress": (0.0, 1.5),
    "FinancialStress": (0.0, 0.5),
    "SocialIsolation": (0.0, -0.5),
    "VictimizationTrauma": (0.0, -1.5),
    "Anxiety": (2.0, 0.0),
    "Depression": (4.0, 0.0),
    "ObservedCovariates": (2.0, -2.15),
}

SEM_EDGES = [
    ("AcademicStress", "Anxiety"),
    ("FinancialStress", "Anxiety"),
    ("SocialIsolation", "Anxiety"),
    ("VictimizationTrauma", "Anxiety"),
    ("AcademicStress", "Depression"),
    ("FinancialStress", "Depression"),
    ("SocialIsolation", "Depression"),
    ("VictimizationTrauma", "Depression"),
    ("Anxiety", "Depression"),
]

OBSERVED_EDGE = ("ObservedCovariates", "Depression")

NOTEBOOK_SIGNIFICANT_SEM_PATHS = [
    {"source": "AcademicStress", "target": "Anxiety", "beta": 0.493560, "p": 1.221245e-14},
    {"source": "SocialIsolation", "target": "Anxiety", "beta": 0.474230, "p": 6.661338e-16},
    {"source": "VictimizationTrauma", "target": "Anxiety", "beta": 0.273283, "p": 5.358472e-03},
    {"source": "Anxiety", "target": "Depression", "beta": 0.744519, "p": 0.0},
    {"source": "FinancialStress", "target": "Depression", "beta": -0.078640, "p": 2.625193e-02},
    {"source": "SocialIsolation", "target": "Depression", "beta": 0.093438, "p": 1.230290e-02},
]

SEM_LATENT_GROUPS = [
    "Anxiety",
    "AcademicStress",
    "FinancialStress",
    "SocialIsolation",
    "VictimizationTrauma",
]

COLOR_POSITIVE = "#c23b3b"
COLOR_NEGATIVE = "#2563a9"
COLOR_NEUTRAL = "#a3a3a3"
COLOR_OUTCOME = "#222222"
PANEL_BACKGROUND = "#f7f8fb"


def first_existing_path(env_name: str, candidates: list[Path]) -> Path | None:
    env_value = os.getenv(env_name)
    if env_value:
        env_path = Path(env_value).expanduser().resolve()
        if env_path.exists():
            return env_path
        raise FileNotFoundError(f"{env_name} points to a missing file: {env_path}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def as_dataframe(value: Any, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame().T
    if isinstance(value, np.ndarray):
        return pd.DataFrame(value)
    raise TypeError(f"{name} must be a pandas DataFrame, Series, or numpy array.")


def load_model_artifact() -> dict[str, Any]:
    artifact_path = first_existing_path("MODEL_ARTIFACT", DEFAULT_ARTIFACT_CANDIDATES)
    if artifact_path is None:
        expected = "\n".join(f"  - {path}" for path in DEFAULT_ARTIFACT_CANDIDATES)
        raise FileNotFoundError(
            "No saved model artifact was found. Save the notebook artifact first.\n"
            f"Checked:\n{expected}"
        )

    artifact = joblib.load(artifact_path)
    if isinstance(artifact, dict):
        model = None
        for key in ("pipeline", "final_pipeline", "model"):
            if key in artifact and artifact[key] is not None:
                model = artifact[key]
                break
        if model is None:
            raise KeyError(
                "Artifact dictionary must contain one of: pipeline, final_pipeline, model."
            )
        return {**artifact, "model": model, "artifact_path": artifact_path}

    return {
        "model": artifact,
        "best_model_name": artifact.__class__.__name__,
        "artifact_path": artifact_path,
    }


def expected_input_features(model: Any) -> list[str] | None:
    if isinstance(model, Pipeline):
        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
            return list(preprocessor.feature_names_in_)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return None


def artifact_list(artifact: dict[str, Any], key: str) -> list[str]:
    value = artifact.get(key, [])
    if value is None:
        return []
    return list(value)


def make_cfa_scorer(artifact: dict[str, Any]) -> Model | None:
    cfa_model_desc = artifact.get("cfa_model_desc") or artifact.get("predictor_cfa_desc")
    cfa_fit_data = artifact.get("cfa_fit_data")
    if cfa_model_desc is None or cfa_fit_data is None:
        return None

    cfa_scorer = Model(cfa_model_desc)
    cfa_scorer.fit(cfa_fit_data)
    return cfa_scorer


def required_raw_features(artifact: dict[str, Any]) -> list[str]:
    return (
        artifact_list(artifact, "predictor_items")
        + artifact_list(artifact, "numeric_observed_features")
        + artifact_list(artifact, "categorical_features")
    )


def can_build_model_input_from_raw(artifact: dict[str, Any], samples: pd.DataFrame) -> bool:
    raw_features = required_raw_features(artifact)
    if not raw_features:
        return False
    return all(feature in samples.columns for feature in raw_features)


def build_model_input_from_raw_artifact(
    raw_input_df: pd.DataFrame,
    artifact: dict[str, Any],
    cfa_scorer: Model | None,
) -> pd.DataFrame:
    if cfa_scorer is None:
        raise ValueError(
            "The artifact does not contain the CFA recipe needed for raw inputs."
        )

    predictor_items = artifact_list(artifact, "predictor_items")
    predictor_latents = artifact_list(artifact, "predictor_latents")
    numeric_observed_features = artifact_list(artifact, "numeric_observed_features")
    categorical_features = artifact_list(artifact, "categorical_features")
    ml_features = artifact_list(artifact, "ml_features")
    cfa_train_medians = artifact.get("cfa_train_medians")

    if not all([predictor_items, predictor_latents, ml_features]):
        raise ValueError("The artifact is missing CFA or model feature metadata.")
    if cfa_train_medians is None:
        raise ValueError("The artifact is missing cfa_train_medians.")

    raw_predictor_items = raw_input_df[predictor_items].apply(
        pd.to_numeric,
        errors="coerce",
    )
    raw_predictor_imp = raw_predictor_items.fillna(cfa_train_medians)

    latent_scores = cfa_scorer.predict_factors(raw_predictor_imp)[predictor_latents]
    latent_scores.columns = [
        f"{latent_name}_score"
        for latent_name in latent_scores.columns
    ]
    latent_scores.index = raw_input_df.index

    observed_features_df = raw_input_df[
        numeric_observed_features + categorical_features
    ].copy()
    model_input = pd.concat([latent_scores, observed_features_df], axis=1)
    return model_input[ml_features]


def normalize_sem_paths(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    sem_paths = artifact.get("sem_paths")
    if sem_paths is None:
        return NOTEBOOK_SIGNIFICANT_SEM_PATHS

    sem_paths_df = as_dataframe(sem_paths, "sem_paths")
    normalized_paths: list[dict[str, Any]] = []
    for _, row in sem_paths_df.iterrows():
        source = row.get("rval") or row.get("source")
        target = row.get("lval") or row.get("target")
        beta = row.get("Estimate", row.get("beta"))
        p_value = row.get("p-value", row.get("p", np.nan))
        if pd.isna(source) or pd.isna(target) or pd.isna(beta):
            continue
        normalized_paths.append(
            {
                "source": str(source),
                "target": str(target),
                "beta": float(beta),
                "p": float(p_value) if not pd.isna(p_value) else np.nan,
            }
        )
    return normalized_paths or NOTEBOOK_SIGNIFICANT_SEM_PATHS


def read_sample_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        try:
            raw = pd.read_excel(path, sheet_name="dataset")
        except ValueError:
            raw = pd.read_excel(path)

        parsed = raw.iloc[2:].copy()
        parsed.columns = raw.iloc[0, :].values
        return parsed

    raise ValueError(f"Unsupported sample file type: {path.suffix}")


def load_samples(
    artifact: dict[str, Any],
    required_features: list[str] | None,
    cfa_scorer: Model | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for key in ("samples", "X_test", "sample_data", "X"):
        if key in artifact and artifact[key] is not None:
            samples = as_dataframe(artifact[key], key)
            break
    else:
        sample_path = first_existing_path("SAMPLE_FILE", DEFAULT_SAMPLE_CANDIDATES)
        if sample_path is None:
            raise FileNotFoundError(
                "No samples were found in the artifact and no SAMPLE_FILE was provided."
            )
        samples = read_sample_file(sample_path)

    display_samples = samples.copy()

    if required_features is not None:
        if all(feature in samples.columns for feature in required_features):
            samples = samples[required_features].copy()
        elif can_build_model_input_from_raw(artifact, samples):
            samples = build_model_input_from_raw_artifact(
                raw_input_df=samples,
                artifact=artifact,
                cfa_scorer=cfa_scorer,
            )
        else:
            missing = [
                feature for feature in required_features
                if feature not in samples.columns
            ]
            missing_preview = ", ".join(missing[:12])
            raise ValueError(
                "Samples are neither model-ready nor raw deployable inputs. "
                f"Missing {len(missing)} model column(s): {missing_preview}."
            )

    model_samples = samples.reset_index(drop=False).rename(columns={"index": "_original_index"})
    display_samples = display_samples.reset_index(drop=False).rename(columns={"index": "_original_index"})
    return model_samples, display_samples


def load_targets(artifact: dict[str, Any], sample_count: int) -> pd.Series | None:
    target_value = None
    for key in ("targets", "y_test", "y"):
        if key in artifact and artifact[key] is not None:
            target_value = artifact[key]
            break
    if target_value is None:
        return None

    targets = pd.Series(target_value).reset_index(drop=True)
    if len(targets) != sample_count:
        return None
    return targets


def split_pipeline_for_shap(model: Any, samples: pd.DataFrame) -> tuple[Any, pd.DataFrame, list[str]]:
    model_inputs = samples.drop(columns=["_original_index"], errors="ignore")

    if isinstance(model, Pipeline) and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        fitted_model = model.named_steps.get("model")
        if fitted_model is None:
            return model, model_inputs, list(model_inputs.columns)

        transformed = preprocessor.transform(model_inputs)
        feature_names = list(preprocessor.get_feature_names_out())
        transformed_df = pd.DataFrame(transformed, columns=feature_names)
        return fitted_model, transformed_df, feature_names

    return model, model_inputs, list(model_inputs.columns)


def build_shap_explainer(explain_model: Any, background: pd.DataFrame) -> shap.Explainer:
    background_sample = background.sample(
        n=min(100, len(background)),
        random_state=42,
    )
    try:
        return shap.Explainer(explain_model, background_sample)
    except Exception:  # noqa: BLE001
        return shap.Explainer(explain_model.predict, background_sample)


def choice_label(position: int, original_index: Any) -> str:
    return f"{position} | original_index={original_index}"


def parse_choice(value: str) -> int:
    return int(str(value).split("|", maxsplit=1)[0].strip())


def infer_feature_group(feature_name: str) -> str:
    feature = feature_name.lower()
    if "anxiety" in feature or "gad" in feature:
        return "Anxiety"
    if "academic" in feature or "ap_luc" in feature:
        return "AcademicStress"
    if "financial" in feature or "tai_chinh" in feature or "chi_phi" in feature:
        return "FinancialStress"
    if "social" in feature or "co_lap" in feature or "bo_roi" in feature or "dong_hanh" in feature:
        return "SocialIsolation"
    if "victim" in feature or "trauma" in feature or "bao_luc" in feature or "xuc_pham" in feature or "de_doa" in feature or "quay_roi" in feature or "xam_hai" in feature:
        return "VictimizationTrauma"
    return "ObservedCovariates"


def edge_contribution(edge: tuple[str, str], group_shap: dict[str, float]) -> float:
    source, target = edge
    if target == "Depression":
        return group_shap.get(source, 0.0)
    if target == "Anxiety":
        anxiety_shap = group_shap.get("Anxiety", 0.0)
        source_shap = group_shap.get(source, 0.0)
        return source_shap if source_shap != 0 else anxiety_shap
    return group_shap.get(source, 0.0)


def display_feature_name(feature_name: str) -> str:
    return feature_name.replace("_score", "")


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}"


def choose_dominant_latent(group_summary: pd.DataFrame) -> str:
    latent_summary = group_summary[group_summary["group"].isin(SEM_LATENT_GROUPS)]
    if latent_summary.empty:
        return "Anxiety"
    return str(latent_summary.sort_values("group_abs", ascending=False).iloc[0]["group"])


def build_bridge_summary(shap_table: pd.DataFrame, sem_paths: list[dict[str, Any]]) -> str:
    top_shap = shap_table.copy()
    top_shap["group"] = top_shap["feature"].astype(str).map(infer_feature_group)
    group_summary = (
        top_shap[top_shap["group"].isin(SEM_LATENT_GROUPS)]
        .groupby("group", as_index=False)
        .agg(group_shap=("shap_value", "sum"), group_abs=("abs_shap", "sum"))
        .sort_values("group_abs", ascending=False)
        .reset_index(drop=True)
    )
    if group_summary.empty:
        return ""

    dominant_group = str(group_summary.iloc[0]["group"])
    dominant_shap = float(group_summary.iloc[0]["group_shap"])
    upstream_sources = [
        path["source"]
        for path in sem_paths
        if path["target"] == dominant_group
    ]
    downstream_targets = [
        path["target"]
        for path in sem_paths
        if path["source"] == dominant_group
    ]

    direction = "increased" if dominant_shap > 0 else "decreased"
    bridge = (
        f"\n\nInterpretation bridge:\n"
        f"- Individual level: {dominant_group} was the strongest latent SHAP driver "
        f"and {direction} this prediction (SHAP {dominant_shap:+.2f})."
    )
    if upstream_sources:
        bridge += (
            f"\n- SEM level: {dominant_group} is structurally associated with "
            f"{', '.join(upstream_sources)} in the population model."
        )
    if downstream_targets:
        bridge += (
            f"\n- SEM level: {dominant_group} also has a significant path to "
            f"{', '.join(downstream_targets)}."
        )
    bridge += "\n- Note: SEM paths are population associations; SHAP values explain this individual prediction."
    return bridge


def plot_shap_contribution_graph(
    shap_table: pd.DataFrame,
    prediction: float,
    base_value: float | None,
    max_display: int,
    sem_paths: list[dict[str, Any]],
    raw_feature_values: pd.Series | None = None,
) -> plt.Figure:
    top_shap = shap_table.head(int(max_display)).copy()
    top_shap["group"] = top_shap["feature"].astype(str).map(infer_feature_group)
    top_shap = top_shap.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    group_summary = (
        top_shap.groupby("group", as_index=False)
        .agg(group_shap=("shap_value", "sum"), group_abs=("abs_shap", "sum"))
        .sort_values("group_abs", ascending=False)
        .reset_index(drop=True)
    )
    group_shap = dict(zip(group_summary["group"], group_summary["group_shap"]))
    group_abs = dict(zip(group_summary["group"], group_summary["group_abs"]))
    dominant_group = choose_dominant_latent(group_summary)

    latent_shap = top_shap[top_shap["group"].isin(SEM_LATENT_GROUPS)].copy()
    if latent_shap.empty:
        latent_shap = top_shap.copy()

    max_abs_group = max(float(group_summary["group_abs"].max()), 1e-9)

    figure_height = max(9, 0.58 * len(latent_shap) + 5)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, figure_height),
        gridspec_kw={"height_ratios": [1.15, 1]},
    )
    figure.patch.set_facecolor("white")

    ax = axes[0]
    ax.set_facecolor(PANEL_BACKGROUND)

    paths_to_draw = [
        path for path in (sem_paths or NOTEBOOK_SIGNIFICANT_SEM_PATHS)
        if path["source"] in SEM_NODE_POSITIONS
        and path["target"] in SEM_NODE_POSITIONS
    ]
    if not paths_to_draw:
        paths_to_draw = NOTEBOOK_SIGNIFICANT_SEM_PATHS
    graph_nodes = list(
        dict.fromkeys(
            [path["source"] for path in paths_to_draw]
            + [path["target"] for path in paths_to_draw]
        )
    )
    pos_sem = SEM_NODE_POSITIONS
    edge_curves = {
        ("AcademicStress", "Anxiety"): 0.08,
        ("SocialIsolation", "Anxiety"): 0.0,
        ("VictimizationTrauma", "Anxiety"): -0.08,
        ("Anxiety", "Depression"): 0.0,
        ("FinancialStress", "Depression"): -0.20,
        ("SocialIsolation", "Depression"): 0.18,
    }
    edge_label_offsets = {
        ("AcademicStress", "Anxiety"): 0.18,
        ("SocialIsolation", "Anxiety"): 0.08,
        ("VictimizationTrauma", "Anxiety"): -0.18,
        ("FinancialStress", "Depression"): -0.25,
        ("SocialIsolation", "Depression"): 0.25,
    }
    max_abs_beta = max(abs(float(path["beta"])) for path in paths_to_draw)
    max_abs_beta = max(max_abs_beta, 1e-9)

    for path in paths_to_draw:
        source = path["source"]
        target = path["target"]
        edge = (source, target)
        source_x, source_y = pos_sem[source]
        target_x, target_y = pos_sem[target]
        beta = float(path["beta"])
        color = COLOR_POSITIVE if beta > 0 else COLOR_NEGATIVE
        edge_width = 1.5 + 4.0 * (abs(beta) / max_abs_beta)
        curve = edge_curves.get(edge, 0.0)

        ax.annotate(
            "",
            xy=(target_x - 0.38, target_y),
            xytext=(source_x + 0.38, source_y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": edge_width,
                "alpha": 0.72,
                "mutation_scale": 20,
                "shrinkA": 4,
                "shrinkB": 4,
                "connectionstyle": f"arc3,rad={curve}",
            },
            zorder=1,
        )

        label_x = (source_x + target_x) / 2
        label_y = (source_y + target_y) / 2 + edge_label_offsets.get(edge, 0.08)
        ax.text(
            label_x,
            label_y,
            f"β = {beta:+.3f}\n{format_p_value(float(path['p']))}",
            color="#111111",
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#d4d4d4", "alpha": 0.92},
        )

    for node in graph_nodes:
        node_x, node_y = pos_sem[node]
        is_dominant = node == dominant_group
        if node == "Depression":
            color = COLOR_OUTCOME
        elif is_dominant:
            color = "#8a5a00"
        else:
            color = "#2f6f95"

        ax.text(
            node_x + 0.035,
            node_y - 0.045,
            SEM_NODE_LABELS[node],
            ha="center",
            va="center",
            fontsize=9,
            color="#333333",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.58,rounding_size=0.18",
                "fc": "#000000",
                "ec": "none",
                "alpha": 0.12,
            },
            zorder=3,
        )
        ax.text(
            node_x,
            node_y,
            SEM_NODE_LABELS[node],
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.58,rounding_size=0.18",
                "fc": color,
                "ec": "#f2c94c" if is_dominant else "white",
                "lw": 2.8 if is_dominant else 1.5,
                "alpha": 0.96,
            },
            zorder=5,
        )

        if is_dominant:
            ax.text(
                node_x,
                node_y - 0.5,
                f"dominant SHAP: {group_shap.get(node, 0.0):+.2f}",
                ha="center",
                va="top",
                fontsize=8,
                color="#333333",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#f2c94c", "alpha": 0.95},
            )

    if base_value is not None:
        ax.text(
            pos_sem["Depression"][0],
            pos_sem["Depression"][1] - 0.58,
            f"base {base_value:.2f}",
            ha="center",
            va="top",
            fontsize=9,
            color="#555555",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#d4d4d4", "alpha": 0.9},
        )

    ax.text(
        2.0,
        1.85,
        f"Dominant individual driver highlighted: {dominant_group}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlim(-0.85, 4.85)
    ax.set_ylim(-2.05, 2.3)
    ax.axis("off")
    ax.set_title("A. Population-Level SEM Paths (β and p-value)", fontsize=13, pad=14)

    ax = axes[1]
    ax.set_facecolor(PANEL_BACKGROUND)
    individual_df = latent_shap.sort_values("shap_value").reset_index(drop=True)
    feature_display = (
        individual_df["feature"]
        .astype(str)
        .map(display_feature_name)
    )
    shap_values = individual_df["shap_value"].to_numpy(dtype=float)
    colors = np.where(shap_values > 0, COLOR_POSITIVE, COLOR_NEGATIVE)

    ax.barh(feature_display, shap_values, color=colors, alpha=0.9)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("SHAP contribution to predicted PHQ")
    ax.set_title(f"B. Individual Prediction — Predicted PHQ = {prediction:.2f}")
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    x_span = max(float(np.max(np.abs(shap_values))) if len(shap_values) else 1.0, 1e-9)
    for y_position, (_, row) in enumerate(individual_df.iterrows()):
        shap_value = float(row["shap_value"])
        feature_name = str(row["feature"])
        if raw_feature_values is not None and feature_name in raw_feature_values.index:
            feature_value = float(raw_feature_values[feature_name])
        else:
            feature_value = float(row["value"])
        offset = 0.035 * x_span if shap_value >= 0 else -0.035 * x_span
        ha = "left" if shap_value >= 0 else "right"
        ax.text(
            shap_value + offset,
            y_position,
            f"SHAP={shap_value:+.2f} | score={feature_value:+.2f}",
            va="center",
            ha=ha,
            fontsize=9,
            color="#262626",
        )

    figure.suptitle(
        "Individual Structural-Predictive Explanation",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def make_app() -> gr.Blocks:
    try:
        artifact = load_model_artifact()
        model = artifact["model"]
        required_features = expected_input_features(model)
        cfa_scorer = make_cfa_scorer(artifact)
        samples, display_samples = load_samples(
            artifact,
            required_features,
            cfa_scorer,
        )
        targets = load_targets(artifact, len(samples))
        explain_model, fallback_background, shap_feature_names = split_pipeline_for_shap(
            model,
            samples,
        )
        if artifact.get("shap_background") is not None:
            shap_background = as_dataframe(
                artifact["shap_background"],
                "shap_background",
            )
        else:
            shap_background = fallback_background
        shap_explainer = build_shap_explainer(explain_model, shap_background)
        sem_paths = normalize_sem_paths(artifact)
        model_name = artifact.get("best_model_name", model.__class__.__name__)
        artifact_path = artifact["artifact_path"]
        choices = [
            choice_label(position, original_index)
            for position, original_index in enumerate(samples["_original_index"].tolist())
        ]
        startup_error = None
    except Exception as exc:  # noqa: BLE001
        artifact = {}
        model = None
        samples = pd.DataFrame()
        display_samples = pd.DataFrame()
        targets = None
        explain_model = None
        shap_background = pd.DataFrame()
        shap_feature_names = []
        shap_explainer = None
        sem_paths = NOTEBOOK_SIGNIFICANT_SEM_PATHS
        model_name = "Unavailable"
        artifact_path = "Unavailable"
        choices = []
        startup_error = exc

    def predict_and_explain(sample_choice: str, max_display: int) -> tuple[str, pd.DataFrame, pd.DataFrame, plt.Figure | None]:
        if startup_error is not None:
            return str(startup_error), pd.DataFrame(), pd.DataFrame(), None

        position = parse_choice(sample_choice)
        row_with_index = samples.iloc[[position]].copy()
        row_for_model = row_with_index.drop(columns=["_original_index"], errors="ignore")
        row_for_display = display_samples.iloc[[position]].copy()

        prediction = float(np.ravel(model.predict(row_for_model))[0])
        actual_text = ""
        if targets is not None:
            actual = float(targets.iloc[position])
            actual_text = f"\nActual PHQ: {actual:.3f}\nError: {prediction - actual:+.3f}"

        if isinstance(model, Pipeline) and "preprocessor" in model.named_steps:
            transformed_row = model.named_steps["preprocessor"].transform(row_for_model)
            shap_row_input = pd.DataFrame(transformed_row, columns=shap_feature_names)
        else:
            shap_row_input = row_for_model

        shap_values = shap_explainer(shap_row_input)
        shap_row = shap_values[0]
        shap_array = np.asarray(shap_row.values).reshape(-1)
        data_array = np.asarray(shap_row.data).reshape(-1)
        base_value_array = np.asarray(shap_row.base_values).reshape(-1)
        base_value = float(base_value_array[0]) if len(base_value_array) else None

        shap_table = pd.DataFrame(
            {
                "feature": shap_feature_names,
                "value": data_array,
                "shap_value": shap_array,
                "abs_shap": np.abs(shap_array),
            }
        ).sort_values("abs_shap", ascending=False, ignore_index=True)

        plt.close("all")
        figure = plot_shap_contribution_graph(
            shap_table=shap_table,
            prediction=prediction,
            base_value=base_value,
            max_display=int(max_display),
            sem_paths=sem_paths,
            raw_feature_values=row_for_model.iloc[0],
        )

        sample_table = row_for_display.drop(
            columns=["_original_index"],
            errors="ignore",
        ).T.reset_index()
        sample_table.columns = ["feature", "value"]

        summary = (
            f"Artifact: {artifact_path}\n"
            f"Model: {model_name}\n"
            f"Selected sample: {row_with_index['_original_index'].iloc[0]}\n"
            f"Predicted PHQ: {prediction:.3f}"
            f"{actual_text}"
            f"{build_bridge_summary(shap_table, sem_paths)}"
        )

        return summary, shap_table.head(int(max_display)), sample_table, figure

    with gr.Blocks(title="PHQ Prediction + SHAP") as demo:
        gr.Markdown("# PHQ Prediction + SHAP")

        if startup_error is not None:
            gr.Markdown(
                "The app could not load the model artifact.\n\n"
                f"```text\n{startup_error}\n```\n\n"
                "Save the notebook artifact first, then run this app again."
            )
        else:
            gr.Markdown(f"Loaded `{model_name}` from `{artifact_path}`.")

            with gr.Row():
                sample_dropdown = gr.Dropdown(
                    choices=choices,
                    value=choices[0],
                    label="Sample",
                )
                max_display_slider = gr.Slider(
                    minimum=5,
                    maximum=30,
                    value=15,
                    step=1,
                    label="Max SHAP features",
                )

            predict_button = gr.Button("Predict and explain", variant="primary")
            summary_output = gr.Textbox(label="Prediction and interpretation", lines=12)
            shap_table_output = gr.Dataframe(label="Top SHAP contributions")
            sample_table_output = gr.Dataframe(label="Selected sample values")
            shap_plot_output = gr.Plot(label="SHAP contribution graph")

            predict_button.click(
                fn=predict_and_explain,
                inputs=[sample_dropdown, max_display_slider],
                outputs=[
                    summary_output,
                    shap_table_output,
                    sample_table_output,
                    shap_plot_output,
                ],
            )
            demo.load(
                fn=predict_and_explain,
                inputs=[sample_dropdown, max_display_slider],
                outputs=[
                    summary_output,
                    shap_table_output,
                    sample_table_output,
                    shap_plot_output,
                ],
            )

    return demo


if __name__ == "__main__":
    app = make_app()
    app.launch()