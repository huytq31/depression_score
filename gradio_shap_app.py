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


def display_feature_name(feature_name: str) -> str:
    return feature_name.replace("_score", "")


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}"


def build_bridge_summary(shap_table: pd.DataFrame, sem_paths: list[dict[str, Any]]) -> str:
    group_summary = aggregate_individual_factors(shap_table)
    group_summary = group_summary[
        group_summary["group"].isin(SEM_LATENT_GROUPS)
    ]
    if group_summary.empty:
        return ""

    dominant = group_summary.iloc[0]
    dominant_group = str(dominant["group"])
    dominant_shap = float(dominant["group_shap"])
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

def aggregate_individual_factors(
    shap_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate feature-level SHAP values into conceptual groups.

    group_shap:
        Net contribution of the construct to this prediction.

    group_abs:
        Total absolute contribution of features belonging to the construct.
    """
    df = shap_table.copy()

    df["group"] = (
        df["feature"]
        .astype(str)
        .map(infer_feature_group)
    )

    summary = (
        df.groupby("group", as_index=False)
        .agg(
            group_shap=("shap_value", "sum"),
            group_abs=("abs_shap", "sum"),
        )
        .sort_values("group_abs", ascending=False)
        .reset_index(drop=True)
    )

    return summary

def get_group_score(
    group: str,
    shap_table: pd.DataFrame,
    raw_feature_values: pd.Series | None,
) -> float | None:
    """
    Try to recover the individual's latent/construct score.
    """

    candidate_names = [
        group,
        f"{group}_score",
    ]

    # First try raw model input
    if raw_feature_values is not None:
        for name in candidate_names:
            if name in raw_feature_values.index:
                try:
                    return float(raw_feature_values[name])
                except (TypeError, ValueError):
                    pass

    # Then try SHAP feature table
    for name in candidate_names:
        match = shap_table[
            shap_table["feature"].astype(str) == name
        ]

        if not match.empty:
            try:
                return float(match.iloc[0]["value"])
            except (TypeError, ValueError):
                pass

    return None

def get_sem_context_for_group(
    group: str,
    sem_paths: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Find population-level SEM relationships surrounding a construct.

    Returns:
        {
            "upstream": [...],
            "downstream": [...]
        }
    """

    upstream = []
    downstream = []

    for path in sem_paths:

        source = str(path["source"])
        target = str(path["target"])

        if target == group:
            upstream.append(path)

        if source == group:
            downstream.append(path)

    upstream = sorted(
        upstream,
        key=lambda x: abs(float(x["beta"])),
        reverse=True,
    )

    downstream = sorted(
        downstream,
        key=lambda x: abs(float(x["beta"])),
        reverse=True,
    )

    return {
        "upstream": upstream,
        "downstream": downstream,
    }

def plot_shap_contribution_graph(
    shap_table: pd.DataFrame,
    prediction: float,
    base_value: float | None,
    max_display: int,
    sem_paths: list[dict[str, Any]],
    raw_feature_values: pd.Series | None = None,
) -> plt.Figure:

    # ============================================================
    # PREPARE
    # ============================================================

    shap_table = shap_table.copy()

    shap_table["group"] = (
        shap_table["feature"]
        .astype(str)
        .map(infer_feature_group)
    )

    group_summary = aggregate_individual_factors(
        shap_table
    )

    latent_summary = group_summary[
        group_summary["group"].isin(
            SEM_LATENT_GROUPS
        )
    ].copy()

    latent_summary = (
        latent_summary
        .sort_values(
            "group_abs",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # show only meaningful latent constructs
    top_groups = latent_summary.head(5)

    if top_groups.empty:
        top_groups = group_summary.head(5)

    dominant_group = (
        str(top_groups.iloc[0]["group"])
        if not top_groups.empty
        else None
    )

    # ============================================================
    # FIGURE
    # ============================================================

    fig = plt.figure(
        figsize=(15, 11),
        facecolor="white",
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[2.3, 1.5],
        hspace=0.35,
    )

    ax = fig.add_subplot(gs[0])
    ax_detail = fig.add_subplot(gs[1])

    # ============================================================
    # A. INDIVIDUAL EXPLANATION GRAPH
    # ============================================================

    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.set_title(
        "Factors Associated With This Student's "
        "Predicted Depression",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # ------------------------------------------------------------
    # Prediction node
    # ------------------------------------------------------------

    prediction_x = 5.0
    prediction_y = 8.8

    ax.text(
        prediction_x,
        prediction_y,
        (
            "Predicted Depression\n"
            f"{prediction:.2f}"
        ),
        ha="center",
        va="center",
        fontsize=15,
        color="white",
        fontweight="bold",
        bbox={
            "boxstyle":
                "round,pad=0.65,rounding_size=0.15",
            "fc": "#222222",
            "ec": "white",
            "lw": 2,
        },
        zorder=10,
    )

    if base_value is not None:

        ax.text(
            prediction_x,
            9.75,
            f"Model baseline = {base_value:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color="#666666",
        )

    # ------------------------------------------------------------
    # Individual factors
    # ------------------------------------------------------------

    n_groups = len(top_groups)

    factor_xs = np.linspace(
        1.0,
        9.0,
        n_groups,
    )

    factor_y = 5.7

    max_abs = max(
        float(top_groups["group_abs"].max()),
        1e-9,
    )

    factor_positions = {}

    for x, (_, row) in zip(
        factor_xs,
        top_groups.iterrows(),
    ):

        group = str(row["group"])

        shap_value = float(
            row["group_shap"]
        )

        group_abs = float(
            row["group_abs"]
        )

        score = get_group_score(
            group,
            shap_table,
            raw_feature_values,
        )

        factor_positions[group] = (
            x,
            factor_y,
        )

        # -----------------------------------------
        # Direction
        # -----------------------------------------

        if shap_value > 0:

            color = COLOR_POSITIVE

            impact_text = (
                f"SHAP {shap_value:+.2f} ↑"
            )

        elif shap_value < 0:

            color = COLOR_NEGATIVE

            impact_text = (
                f"SHAP {shap_value:+.2f} ↓"
            )

        else:

            color = COLOR_NEUTRAL

            impact_text = (
                f"SHAP {shap_value:+.2f}"
            )

        if score is not None:

            score_text = (
                f"Factor score {score:+.2f}"
            )

        else:

            score_text = ""

        label = (
            f"{SEM_NODE_LABELS.get(group, group)}\n"
            f"{score_text}\n"
            f"{impact_text}"
        )

        relative_strength = (
            group_abs / max_abs
        )

        # -----------------------------------------
        # Factor node
        # -----------------------------------------

        ax.text(
            x,
            factor_y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
            bbox={
                "boxstyle":
                    "round,pad=0.45,"
                    "rounding_size=0.12",
                "fc": color,
                "ec": (
                    "#f2c94c"
                    if group == dominant_group
                    else "white"
                ),
                "lw": (
                    3
                    if group == dominant_group
                    else 1.5
                ),
            },
            zorder=5,
        )

        # -----------------------------------------
        # Factor -> Prediction
        #
        # IMPORTANT:
        # thickness represents INDIVIDUAL SHAP
        # NOT SEM beta.
        # -----------------------------------------

        edge_width = (
            0.8
            + 6.0 * relative_strength
        )

        ax.annotate(
            "",
            xy=(
                prediction_x,
                prediction_y - 0.65,
            ),
            xytext=(
                x,
                factor_y + 0.65,
            ),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": edge_width,
                "alpha": 0.65,
                "mutation_scale": 15,
            },
            zorder=1,
        )

    # ------------------------------------------------------------
    # Labels explaining SHAP edges
    # ------------------------------------------------------------

    ax.text(
        5.0,
        7.25,
        "Individual contribution to prediction (SHAP)",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
        style="italic",
    )

    # ============================================================
    # SEM CONTEXT AROUND DOMINANT FACTOR ONLY
    # ============================================================

    if dominant_group is not None:

        context = get_sem_context_for_group(
            dominant_group,
            sem_paths,
        )

        upstream = context["upstream"]

        dominant_x, dominant_y = (
            factor_positions[
                dominant_group
            ]
        )

        # Maximum 4 upstream SEM factors
        upstream = upstream[:4]

        if upstream:

            context_y = 2.0

            context_xs = np.linspace(
                max(0.8, dominant_x - 2.0),
                min(9.2, dominant_x + 2.0),
                len(upstream),
            )

            for context_x, path in zip(
                context_xs,
                upstream,
            ):

                source = str(
                    path["source"]
                )

                beta = float(
                    path["beta"]
                )

                p_value = float(
                    path.get(
                        "p",
                        np.nan,
                    )
                )

                source_label = (
                    SEM_NODE_LABELS.get(
                        source,
                        source,
                    )
                )

                ax.text(
                    context_x,
                    context_y,
                    source_label,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white",
                    fontweight="bold",
                    bbox={
                        "boxstyle":
                            "round,pad=0.35",
                        "fc": "#64748b",
                        "ec": "white",
                    },
                    zorder=5,
                )

                # SEM relationship
                ax.annotate(
                    "",
                    xy=(
                        dominant_x,
                        dominant_y - 0.65,
                    ),
                    xytext=(
                        context_x,
                        context_y + 0.45,
                    ),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#888888",
                        "lw":
                            1.2
                            + 2.0 * abs(beta),
                        "alpha": 0.55,
                        "mutation_scale": 13,
                    },
                    zorder=1,
                )

                mid_x = (
                    context_x
                    + dominant_x
                ) / 2

                mid_y = (
                    context_y
                    + dominant_y
                ) / 2

                if np.isfinite(p_value):

                    beta_label = (
                        f"β={beta:+.2f}\n"
                        f"{format_p_value(p_value)}"
                    )

                else:

                    beta_label = (
                        f"β={beta:+.2f}"
                    )

                ax.text(
                    mid_x,
                    mid_y,
                    beta_label,
                    fontsize=7,
                    ha="center",
                    va="center",
                    color="#555555",
                    bbox={
                        "boxstyle":
                            "round,pad=0.15",
                        "fc": "white",
                        "ec": "#dddddd",
                        "alpha": 0.90,
                    },
                )

            ax.text(
                dominant_x,
                0.75,
                (
                    "Population-level context around "
                    f"{SEM_NODE_LABELS.get(dominant_group, dominant_group).replace(chr(10), ' ')}"
                ),
                ha="center",
                fontsize=8.5,
                color="#666666",
                style="italic",
            )

    # ============================================================
    # B. DETAILED INDIVIDUAL CONTRIBUTIONS
    # ============================================================

    ax_detail.set_facecolor(
        PANEL_BACKGROUND
    )

    top_features = (
        shap_table
        .sort_values(
            "abs_shap",
            ascending=False,
        )
        .head(int(max_display))
        .sort_values(
            "shap_value",
            ascending=True,
        )
    )

    values = top_features[
        "shap_value"
    ].to_numpy(dtype=float)

    labels = (
        top_features["feature"]
        .astype(str)
        .map(display_feature_name)
        .tolist()
    )

    colors = [
        (
            COLOR_POSITIVE
            if value >= 0
            else COLOR_NEGATIVE
        )
        for value in values
    ]

    y = np.arange(
        len(top_features)
    )

    ax_detail.barh(
        y,
        values,
        color=colors,
        alpha=0.9,
    )

    ax_detail.set_yticks(y)

    ax_detail.set_yticklabels(
        labels,
        fontsize=8.5,
    )

    ax_detail.axvline(
        0,
        color="#333333",
        lw=1,
    )

    ax_detail.grid(
        axis="x",
        linestyle="--",
        alpha=0.18,
    )

    ax_detail.spines[
        ["top", "right", "left"]
    ].set_visible(False)

    ax_detail.tick_params(
        axis="y",
        length=0,
    )

    ax_detail.set_xlabel(
        "SHAP contribution to predicted depression"
    )

    ax_detail.set_title(
        "Detailed Individual Contributions",
        fontsize=12,
        fontweight="bold",
    )

    # ------------------------------------------------------------
    # SHAP labels
    # ------------------------------------------------------------

    max_abs_feature = max(
        np.max(np.abs(values))
        if len(values)
        else 1.0,
        1e-9,
    )

    offset = (
        0.025 * max_abs_feature
    )

    for yy, value in zip(
        y,
        values,
    ):

        if value >= 0:
            xx = value + offset
            ha = "left"
        else:
            xx = value - offset
            ha = "right"

        ax_detail.text(
            xx,
            yy,
            f"{value:+.3f}",
            va="center",
            ha=ha,
            fontsize=8,
            fontweight="bold",
        )

    # ============================================================
    # FOOTNOTE
    # ============================================================

    fig.text(
        0.5,
        0.015,
        (
            "SHAP = contribution to this student's model prediction. "
            "β = population-level SEM association. "
            "These quantities describe model contributions and "
            "associations, not causal effects."
        ),
        ha="center",
        fontsize=8,
        color="#666666",
        style="italic",
    )

    fig.subplots_adjust(
        top=0.92,
        bottom=0.07,
        left=0.12,
        right=0.96,
    )

    return fig


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