"""Reproducible analysis for the locked R1 11-model transfer-learning experiment.

This module is deliberately read-only with respect to the final SQLite result store. It
separates seed-level stability reporting from patient-level reporting, so 100 stochastic
training seeds are never treated as 100 independent clinical samples.
"""
from __future__ import annotations

import json
import math
import sqlite3
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Union

import matplotlib

# Safe for scripts and notebooks running on a server without a graphical display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

SUPPORTED_MODELS: tuple[str, ...] = (
    "VGG16",
    "VGG19",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "InceptionV3",
    "InceptionResNetV2",
    "DenseNet169",
    "DenseNet201",
    "MobileNetV2",
    "Xception",
)

METRIC_COLUMNS: Mapping[str, str] = {
    "auc": "test_auc",
    "accuracy": "test_accuracy",
    "sensitivity": "test_sensitivity",
    "specificity": "test_specificity",
    "precision": "test_precision",
    "f1": "test_f1",
}
CONFUSION_COLUMNS: tuple[str, ...] = ("test_tp", "test_tn", "test_fp", "test_fn")
DISPLAY_NAMES: Mapping[str, str] = {
    "auc": "AUC",
    "accuracy": "Accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "f1": "F1 score",
    "test_tp": "TP",
    "test_tn": "TN",
    "test_fp": "FP",
    "test_fn": "FN",
}
PREPROCESSING_DESCRIPTIONS: Mapping[str, str] = {
    "VGG16": "Keras VGG preprocess_input: RGB→BGR; ImageNet mean-centering.",
    "VGG19": "Keras VGG preprocess_input: RGB→BGR; ImageNet mean-centering.",
    "ResNet50": "Keras ResNet preprocess_input: RGB→BGR; ImageNet mean-centering.",
    "ResNet101": "Keras ResNet preprocess_input: RGB→BGR; ImageNet mean-centering.",
    "ResNet152": "Keras ResNet preprocess_input: RGB→BGR; ImageNet mean-centering.",
    "InceptionV3": "Keras Inception preprocess_input: scales RGB pixels from 0–255 to −1 to +1.",
    "InceptionResNetV2": "Keras Inception-ResNet preprocess_input: scales RGB pixels from 0–255 to −1 to +1.",
    "DenseNet169": "Keras DenseNet preprocess_input: ImageNet RGB channel normalization.",
    "DenseNet201": "Keras DenseNet preprocess_input: ImageNet RGB channel normalization.",
    "MobileNetV2": "Keras MobileNetV2 preprocess_input: scales RGB pixels from 0–255 to −1 to +1.",
    "Xception": "Keras Xception preprocess_input: scales RGB pixels from 0–255 to −1 to +1.",
}


@dataclass(frozen=True)
class AnalysisSettings:
    """Locked choices for this analysis pass; no result-dependent selection occurs."""

    aggregation_rule: str = "mean_probability_across_100_seeds"
    threshold: float = 0.50
    metric_bootstrap_iterations: int = 2000
    dca_bootstrap_iterations: int = 1000
    random_seed: int = 20260926
    calibration_bins: int = 5


@dataclass
class R1Results:
    """Read-only in-memory view of the secure final result store."""

    run_status: pd.DataFrame
    run_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    phase_parameters: pd.DataFrame
    batches: pd.DataFrame
    metadata: dict[str, Any]


# ---------- Database loading and integrity ----------

def _query(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table_name}", connection)


def load_r1_results(results_dir: Union[Path, str]) -> R1Results:
    """Read the final SQLite store without making any write or schema operation."""
    results_dir = Path(results_dir)
    database = results_dir / "r1_final_tl_runs.sqlite"
    if not database.exists():
        raise FileNotFoundError(f"No final SQLite results database at {database}")
    uri = f"file:{database.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        result = R1Results(
            run_status=_query(connection, "run_status").sort_values(["model_name", "seed"]),
            run_metrics=_query(connection, "run_metrics").sort_values(["model_name", "seed"]),
            test_predictions=_query(connection, "test_predictions").sort_values(["model_name", "seed", "patient_id"]),
            phase_parameters=_query(connection, "phase_parameters").sort_values(["model_name", "seed", "phase"]),
            batches=_query(connection, "batches").sort_values("created_at"),
            metadata={
                row["key"]: json.loads(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata ORDER BY key").fetchall()
            },
        )
    finally:
        connection.close()
    validate_r1_results(result)
    return result


def validate_r1_results(result: R1Results, expected_models: Sequence[str] = SUPPORTED_MODELS, expected_seeds: Sequence[int] = tuple(range(1, 101))) -> None:
    """Fail early if the result store is not the complete locked 11×100 experiment."""
    expected_grid = {(model, seed) for model in expected_models for seed in expected_seeds}
    completed = result.run_status.loc[result.run_status["status"] == "completed", ["model_name", "seed"]]
    observed_grid = set(map(tuple, completed.itertuples(index=False, name=None)))
    if observed_grid != expected_grid:
        missing = sorted(expected_grid - observed_grid)
        unexpected = sorted(observed_grid - expected_grid)
        raise ValueError(f"Final result store is not the expected complete 11×100 grid; missing={missing[:5]}, unexpected={unexpected[:5]}")
    if result.run_status["status"].ne("completed").any():
        raise ValueError("Final result store has non-completed run-status records")
    if result.run_metrics.duplicated(["model_name", "seed"]).any():
        raise ValueError("Duplicate run metric record")
    if len(result.run_metrics) != len(expected_grid):
        raise ValueError("Metric records do not cover the expected run grid")
    predictions_per_run = result.test_predictions.groupby(["model_name", "seed"]).size()
    if len(predictions_per_run) != len(expected_grid) or not predictions_per_run.eq(46).all():
        raise ValueError("Every completed run must have exactly 46 test predictions")
    if result.test_predictions.duplicated(["patient_id", "model_name", "seed"]).any():
        raise ValueError("Duplicate patient/model/seed prediction record")
    if not result.test_predictions["probability"].between(0, 1).all():
        raise ValueError("Test probabilities must lie between 0 and 1")
    phases_per_run = result.phase_parameters.groupby(["model_name", "seed"]).size()
    if len(phases_per_run) != len(expected_grid) or not phases_per_run.eq(3).all():
        raise ValueError("Every completed run must have exactly three phase records")
    label_counts = result.test_predictions.groupby("patient_id")["observed_label"].nunique()
    if not label_counts.eq(1).all():
        raise ValueError("A test patient has inconsistent labels across runs")


def _batch_manifests(result: R1Results) -> list[dict[str, Any]]:
    manifests = []
    for row in result.batches.itertuples(index=False):
        payload = json.loads(row.manifest_json)
        payload["created_at"] = row.created_at
        manifests.append(payload)
    return manifests


def integrity_table(result: R1Results) -> pd.DataFrame:
    """Compact record of the locked protocol and structural completion."""
    manifests = _batch_manifests(result)
    first_manifest = manifests[0] if manifests else {}
    protocol_columns = [
        "protocol_id", "head_learning_rate", "fine_tune_learning_rate", "dropout_rate",
        "input_size", "batch_size", "optimizer", "loss", "class_weights", "augmentation",
        "threshold", "phase1_epochs", "phase2_epochs", "phase3_epochs", "early_stopping_patience",
    ]
    unique_protocol = result.run_metrics[protocol_columns].drop_duplicates()
    if len(unique_protocol) != 1:
        raise ValueError("More than one protocol is present in the final result store")
    protocol = unique_protocol.iloc[0].to_dict()
    split_counts = first_manifest.get("split_counts", {})
    class_counts = first_manifest.get("split_class_counts", {})
    rows = [
        ("Structural completion", f"{len(result.run_metrics)} completed runs; 11 architectures × 100 prespecified seeds"),
        ("Locked split", f"train={split_counts.get('train', 'n/a')}; validation={split_counts.get('validation', 'n/a')}; test={split_counts.get('test', 'n/a')} patients"),
        ("Training class counts", str(class_counts.get("train", "n/a"))),
        ("Validation class counts", str(class_counts.get("validation", "n/a"))),
        ("Test class counts", str(class_counts.get("test", "n/a"))),
        ("Protocol ID", str(protocol["protocol_id"])),
        ("Input", f"{protocol['input_size']} × {protocol['input_size']} RGB JPEG polar maps"),
        ("Optimizer / loss", f"{protocol['optimizer']} / {protocol['loss']}"),
        ("Learning rates", f"head={protocol['head_learning_rate']:.0e}; fine-tuning={protocol['fine_tune_learning_rate']:.0e}"),
        ("Dropout / batch size", f"dropout={protocol['dropout_rate']:.2f}; batch size={int(protocol['batch_size'])}"),
        ("Augmentation / class weighting", f"augmentation={protocol['augmentation']}; class weights={bool(protocol['class_weights'])}"),
        ("Threshold", f"{protocol['threshold']:.2f}"),
        ("Test-set selection", "No model/protocol/seed selection was performed from test results."),
        ("Repeated seeds", "100 runs are reported as descriptive stochastic-training stability, not independent clinical samples."),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def preprocessing_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"model_name": model, "keras_preprocessing": PREPROCESSING_DESCRIPTIONS[model]} for model in SUPPORTED_MODELS]
    )


# ---------- Seed-level descriptive stability ----------

def _quantile(series: pd.Series, probability: float) -> float:
    return float(series.quantile(probability))


def seed_stability_long(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Descriptive seed-level results; this function performs no hypothesis testing."""
    rows: list[dict[str, Any]] = []
    for model_name, group in run_metrics.groupby("model_name", sort=True):
        for metric, column in METRIC_COLUMNS.items():
            values = group[column].astype(float)
            median = float(values.median())
            q1, q3 = _quantile(values, 0.25), _quantile(values, 0.75)
            rows.append({
                "model_name": model_name,
                "metric": DISPLAY_NAMES[metric],
                "n_seeds": int(len(values)),
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "display_median_iqr": f"{median:.3f} ({q3 - q1:.3f})",
            })
    return pd.DataFrame(rows)


def seed_stability_main_table(run_metrics: pd.DataFrame) -> pd.DataFrame:
    long = seed_stability_long(run_metrics)
    table = long.pivot(index="model_name", columns="metric", values="display_median_iqr").reset_index()
    desired = ["model_name", "AUC", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1 score"]
    return table.reindex(columns=desired).sort_values("model_name").reset_index(drop=True)


def seed_confusion_long(run_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, group in run_metrics.groupby("model_name", sort=True):
        for column in CONFUSION_COLUMNS:
            values = group[column].astype(float)
            median = float(values.median())
            q1, q3 = _quantile(values, 0.25), _quantile(values, 0.75)
            rows.append({
                "model_name": model_name,
                "metric": DISPLAY_NAMES[column],
                "n_seeds": int(len(values)),
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "display_median_iqr": f"{median:.1f} ({q3 - q1:.1f})",
            })
    return pd.DataFrame(rows)


def seed_confusion_table(run_metrics: pd.DataFrame) -> pd.DataFrame:
    long = seed_confusion_long(run_metrics)
    table = long.pivot(index="model_name", columns="metric", values="display_median_iqr").reset_index()
    return table.reindex(columns=["model_name", "TP", "TN", "FP", "FN"]).sort_values("model_name").reset_index(drop=True)


def runtime_table(run_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        run_metrics.groupby("model_name", as_index=False)["elapsed_seconds"]
        .agg(n_seeds="size", median_seconds="median", q1_seconds=lambda x: x.quantile(0.25), q3_seconds=lambda x: x.quantile(0.75), total_seconds="sum")
        .assign(iqr_seconds=lambda x: x.q3_seconds - x.q1_seconds, median_minutes=lambda x: x.median_seconds / 60.0, total_hours=lambda x: x.total_seconds / 3600.0)
        .sort_values("model_name")
        .reset_index(drop=True)
    )


def phase_audit_table(phase_parameters: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, phase), group in phase_parameters.groupby(["model_name", "phase"], sort=True):
        actual = group["actual_epochs"].astype(float)
        rows.append({
            "model_name": model_name,
            "phase": int(phase),
            "n_seeds": int(len(group)),
            "total_params": int(group["total_params"].iloc[0]),
            "trainable_params": int(group["trainable_params"].iloc[0]),
            "non_trainable_params": int(group["non_trainable_params"].iloc[0]),
            "selected_backbone_layers": int(group["selected_backbone_layers"].iloc[0]),
            "learning_rate": float(group["learning_rate"].iloc[0]),
            "requested_epochs": int(group["requested_epochs"].iloc[0]),
            "actual_epochs_median": float(actual.median()),
            "actual_epochs_q1": _quantile(actual, 0.25),
            "actual_epochs_q3": _quantile(actual, 0.75),
            "actual_epochs_iqr": _quantile(actual, 0.75) - _quantile(actual, 0.25),
        })
    return pd.DataFrame(rows).sort_values(["model_name", "phase"]).reset_index(drop=True)


# ---------- Locked patient-level reporting ----------

def aggregate_patient_probabilities(test_predictions: pd.DataFrame, settings: AnalysisSettings) -> pd.DataFrame:
    """Create one final test probability per patient/model using the declared mean rule."""
    if settings.aggregation_rule != "mean_probability_across_100_seeds":
        raise ValueError(f"Unsupported aggregation rule: {settings.aggregation_rule}")
    grouped = (
        test_predictions.groupby(["model_name", "patient_id", "observed_label"], as_index=False)["probability"]
        .agg(seed_probability_mean="mean", seed_probability_median="median", seed_probability_sd="std", n_seeds="size")
    )
    if not grouped["n_seeds"].eq(100).all():
        raise ValueError("Every patient/model must have 100 seed probabilities before aggregation")
    grouped["final_probability"] = grouped["seed_probability_mean"]
    grouped["threshold"] = settings.threshold
    grouped["final_binary_prediction"] = (grouped["final_probability"] >= settings.threshold).astype(int)
    return grouped.sort_values(["model_name", "patient_id"]).reset_index(drop=True)


def _binary_performance(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    binary = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, binary, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, binary)),
        "sensitivity": float(recall_score(y_true, binary, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y_true, binary, zero_division=0)),
        "f1": float(f1_score(y_true, binary, zero_division=0)),
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
    }


def _stratified_resample_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = np.arange(len(y_true))
    positive = positions[y_true == 1]
    negative = positions[y_true == 0]
    return np.concatenate([
        rng.choice(positive, size=len(positive), replace=True),
        rng.choice(negative, size=len(negative), replace=True),
    ])


def patient_metric_table(aggregated: pd.DataFrame, settings: AnalysisSettings) -> pd.DataFrame:
    """Patient-level performance with stratified nonparametric bootstrap percentile CIs."""
    rng = np.random.default_rng(settings.random_seed)
    rows: list[dict[str, Any]] = []
    metrics = ("auc", "accuracy", "sensitivity", "specificity", "precision", "f1")
    for model_name, group in aggregated.groupby("model_name", sort=True):
        y_true = group["observed_label"].to_numpy(dtype=int)
        probabilities = group["final_probability"].to_numpy(dtype=float)
        point = _binary_performance(y_true, probabilities, settings.threshold)
        boot = {metric: np.empty(settings.metric_bootstrap_iterations, dtype=float) for metric in metrics}
        for index in range(settings.metric_bootstrap_iterations):
            draw = _stratified_resample_indices(y_true, rng)
            values = _binary_performance(y_true[draw], probabilities[draw], settings.threshold)
            for metric in metrics:
                boot[metric][index] = values[metric]
        row: dict[str, Any] = {"model_name": model_name, "n_patients": int(len(group)), "threshold": settings.threshold}
        for metric in metrics:
            lower, upper = np.quantile(boot[metric], [0.025, 0.975])
            row[f"{metric}_point"] = point[metric]
            row[f"{metric}_ci_low"] = float(lower)
            row[f"{metric}_ci_high"] = float(upper)
            row[f"{metric}_display"] = f"{point[metric]:.3f} ({lower:.3f}–{upper:.3f})"
        for item in ("tn", "fp", "fn", "tp"):
            row[item] = int(point[item])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def final_confusion_table(aggregated: pd.DataFrame, settings: AnalysisSettings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, group in aggregated.groupby("model_name", sort=True):
        values = _binary_performance(
            group["observed_label"].to_numpy(dtype=int),
            group["final_probability"].to_numpy(dtype=float),
            settings.threshold,
        )
        rows.append({"model_name": model_name, "TP": int(values["tp"]), "TN": int(values["tn"]), "FP": int(values["fp"]), "FN": int(values["fn"])})
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def calibration_summary(aggregated: pd.DataFrame, settings: AnalysisSettings) -> pd.DataFrame:
    """Brier score, expected calibration error, and logistic calibration intercept/slope."""
    rows: list[dict[str, Any]] = []
    for model_name, group in aggregated.groupby("model_name", sort=True):
        y_true = group["observed_label"].to_numpy(dtype=int)
        probability = group["final_probability"].to_numpy(dtype=float)
        bins = pd.cut(probability, bins=np.linspace(0, 1, settings.calibration_bins + 1), include_lowest=True)
        bin_frame = pd.DataFrame({"bin": bins, "y": y_true, "p": probability}).groupby("bin", observed=False)
        ece = 0.0
        for _, part in bin_frame:
            if len(part):
                ece += len(part) / len(group) * abs(part["y"].mean() - part["p"].mean())
        intercept, slope = _calibration_intercept_slope(y_true, probability)
        rows.append({
            "model_name": model_name,
            "n_patients": int(len(group)),
            "brier_score": float(brier_score_loss(y_true, probability)),
            "expected_calibration_error": float(ece),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "mean_predicted_probability": float(probability.mean()),
            "observed_prevalence": float(y_true.mean()),
        })
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def _calibration_intercept_slope(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    logit_probability = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 'none' is accepted by the hospital environment's scikit-learn 1.3.x;
            # this auxiliary calibration regression intentionally has no penalty.
            model = LogisticRegression(penalty="none", fit_intercept=True, solver="lbfgs", max_iter=1000)
            model.fit(logit_probability.reshape(-1, 1), y_true)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return float("nan"), float("nan")


# ---------- Figures ----------

def _figure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.titleweight": "bold"})


def plot_stability_boxplots(run_metrics: pd.DataFrame, output: Path) -> None:
    _figure_style()
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for axis, (metric, column) in zip(axes.ravel(), METRIC_COLUMNS.items()):
        data = run_metrics[["model_name", column]].rename(columns={column: "value"})
        order = data.groupby("model_name")["value"].median().sort_values(ascending=False).index
        sns.boxplot(data=data, x="model_name", y="value", order=order, color="#78A6C8", ax=axis, fliersize=2)
        sns.stripplot(data=data, x="model_name", y="value", order=order, color="#1F4E78", alpha=0.25, size=2, ax=axis)
        axis.set_title(DISPLAY_NAMES[metric] + " across 100 seeds")
        axis.set_xlabel("")
        axis.set_ylabel(DISPLAY_NAMES[metric])
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylim(0, 1.05)
    figure.suptitle("Seed-level stability (descriptive; not patient-level inference)", y=1.02, fontsize=16, fontweight="bold")
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_sensitivity_specificity(run_metrics: pd.DataFrame, output: Path) -> None:
    _figure_style()
    summary = run_metrics.groupby("model_name")[["test_sensitivity", "test_specificity"]].median().reset_index()
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.scatter(1 - summary["test_specificity"], summary["test_sensitivity"], s=75, color="#1F4E78")
    for row in summary.itertuples(index=False):
        axis.annotate(row.model_name, (1 - row.test_specificity, row.test_sensitivity), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("False-positive rate (1 − specificity)")
    axis.set_ylabel("Sensitivity")
    axis.set_title("Sensitivity–specificity trade-off\n(median across 100 seeds; descriptive)")
    axis.set_xlim(-0.02, max(0.35, (1 - summary["test_specificity"]).max() + 0.04))
    axis.set_ylim(0, 1.05)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_tradeoff(run_metrics: pd.DataFrame, output: Path) -> None:
    _figure_style()
    summary = run_metrics.groupby("model_name")[["test_fp", "test_fn"]].median().reset_index()
    summary = summary.sort_values(["test_fn", "test_fp"], ascending=False)
    y = np.arange(len(summary))
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(y - 0.18, summary["test_fn"], height=0.36, color="#D55E00", label="False negatives")
    axis.barh(y + 0.18, summary["test_fp"], height=0.36, color="#0072B2", label="False positives")
    axis.set_yticks(y, summary["model_name"])
    axis.set_xlabel("Median number of test patients across 100 seeds")
    axis.set_title("False-negative / false-positive trade-off (descriptive)")
    axis.legend()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_runtime_by_model(run_metrics: pd.DataFrame, output: Path) -> None:
    _figure_style()
    summary = runtime_table(run_metrics).sort_values("median_seconds", ascending=True)
    figure, axis = plt.subplots(figsize=(10, 7))
    y = np.arange(len(summary))
    axis.errorbar(
        summary["median_seconds"] / 60.0,
        y,
        xerr=[
            (summary["median_seconds"] - summary["q1_seconds"]) / 60.0,
            (summary["q3_seconds"] - summary["median_seconds"]) / 60.0,
        ],
        fmt="o",
        color="#1F4E78",
        ecolor="#78A6C8",
        capsize=3,
    )
    axis.set_yticks(y, summary["model_name"])
    axis.set_xlabel("Elapsed time per seed (minutes; median and IQR)")
    axis.set_title("Observed R1 training runtime by architecture")
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_phase_epochs(phase_parameters: pd.DataFrame, output: Path) -> None:
    _figure_style()
    summary = phase_audit_table(phase_parameters)
    figure, axis = plt.subplots(figsize=(12, 7))
    palette = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B"}
    for phase in (1, 2, 3):
        part = summary[summary["phase"] == phase]
        axis.scatter(part["model_name"], part["actual_epochs_median"], s=55, color=palette[phase], label=f"Phase {phase}")
    axis.axhline(30, color="black", linestyle="--", lw=1, label="Maximum requested epochs")
    axis.set_ylabel("Actual epochs completed (median across 100 seeds)")
    axis.set_xlabel("")
    axis.set_title("Early-stopping duration by phase and architecture")
    axis.tick_params(axis="x", rotation=45)
    axis.legend(ncol=2)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_roc(aggregated: pd.DataFrame, output: Path) -> None:
    _figure_style()
    figure, axis = plt.subplots(figsize=(9, 8))
    palette = sns.color_palette("tab20", n_colors=aggregated["model_name"].nunique())
    for colour, (model_name, group) in zip(palette, aggregated.groupby("model_name", sort=True)):
        fpr, tpr, _ = roc_curve(group["observed_label"], group["final_probability"])
        auc = roc_auc_score(group["observed_label"], group["final_probability"])
        axis.plot(fpr, tpr, lw=1.8, color=colour, label=f"{model_name} (AUC {auc:.3f})")
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1, label="Chance")
    axis.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves from locked seed-aggregated test probabilities")
    axis.legend(loc="lower right", fontsize=8, frameon=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_calibration(aggregated: pd.DataFrame, settings: AnalysisSettings, output: Path) -> None:
    _figure_style()
    figure, axis = plt.subplots(figsize=(9, 8))
    palette = sns.color_palette("tab20", n_colors=aggregated["model_name"].nunique())
    for colour, (model_name, group) in zip(palette, aggregated.groupby("model_name", sort=True)):
        observed, predicted = calibration_curve(group["observed_label"], group["final_probability"], n_bins=settings.calibration_bins, strategy="uniform")
        axis.plot(predicted, observed, marker="o", lw=1.2, markersize=4, color=colour, label=model_name)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", lw=1, label="Perfect calibration")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed proportion positive", title=f"Calibration on locked test predictions ({settings.calibration_bins} uniform bins)")
    axis.legend(loc="upper left", fontsize=8, ncol=2, frameon=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _net_benefit(y_true: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    values = []
    n = len(y_true)
    for threshold in thresholds:
        prediction = probability >= threshold
        tp = np.sum((prediction == 1) & (y_true == 1))
        fp = np.sum((prediction == 1) & (y_true == 0))
        values.append(tp / n - fp / n * threshold / (1 - threshold))
    return np.asarray(values, dtype=float)


def decision_curve_data(aggregated: pd.DataFrame, settings: AnalysisSettings) -> pd.DataFrame:
    """Return decision-curve point estimates and stratified-bootstrap uncertainty bands."""
    thresholds = np.arange(0.05, 0.96, 0.01)
    rng = np.random.default_rng(settings.random_seed + 17)
    rows: list[dict[str, Any]] = []
    for model_name, group in aggregated.groupby("model_name", sort=True):
        y_true = group["observed_label"].to_numpy(dtype=int)
        probability = group["final_probability"].to_numpy(dtype=float)
        point = _net_benefit(y_true, probability, thresholds)
        boot = np.empty((settings.dca_bootstrap_iterations, len(thresholds)), dtype=float)
        for index in range(settings.dca_bootstrap_iterations):
            draw = _stratified_resample_indices(y_true, rng)
            boot[index] = _net_benefit(y_true[draw], probability[draw], thresholds)
        lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
        for threshold, estimate, low, high in zip(thresholds, point, lower, upper):
            rows.append({"model_name": model_name, "threshold_probability": float(threshold), "net_benefit": float(estimate), "ci_low": float(low), "ci_high": float(high)})
    prevalence = float(aggregated.drop_duplicates("patient_id")["observed_label"].mean())
    for threshold in thresholds:
        treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        rows.append({"model_name": "Treat all", "threshold_probability": float(threshold), "net_benefit": float(treat_all), "ci_low": float("nan"), "ci_high": float("nan")})
        rows.append({"model_name": "Treat none", "threshold_probability": float(threshold), "net_benefit": 0.0, "ci_low": float("nan"), "ci_high": float("nan")})
    return pd.DataFrame(rows)


def plot_dca_overview(dca: pd.DataFrame, output: Path) -> None:
    _figure_style()
    figure, axis = plt.subplots(figsize=(10, 8))
    for model_name, group in dca.groupby("model_name", sort=True):
        if model_name in {"Treat all", "Treat none"}:
            style = "--" if model_name == "Treat all" else ":"
            axis.plot(group["threshold_probability"], group["net_benefit"], color="black", linestyle=style, lw=1.2, label=model_name)
        else:
            axis.plot(group["threshold_probability"], group["net_benefit"], lw=1.3, label=model_name)
    axis.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision-curve analysis: locked seed-aggregated test probabilities")
    axis.legend(fontsize=7, ncol=2, loc="upper right")
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_dca_uncertainty(dca: pd.DataFrame, output: Path) -> None:
    _figure_style()
    models = [model for model in sorted(dca["model_name"].unique()) if model not in {"Treat all", "Treat none"}]
    figure, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=True, sharey=True, constrained_layout=True)
    for axis, model_name in zip(axes.ravel(), models):
        group = dca[dca["model_name"] == model_name]
        axis.plot(group["threshold_probability"], group["net_benefit"], color="#1F4E78", lw=1.5)
        axis.fill_between(group["threshold_probability"], group["ci_low"], group["ci_high"], color="#78A6C8", alpha=0.35)
        axis.axhline(0, color="black", lw=0.7, linestyle=":")
        axis.set_title(model_name, fontsize=10)
    for axis in axes.ravel()[len(models):]:
        axis.axis("off")
    figure.supxlabel("Threshold probability")
    figure.supylabel("Net benefit")
    figure.suptitle("Decision-curve uncertainty: 95% stratified-bootstrap bands", fontsize=15, fontweight="bold")
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ---------- End-to-end local export ----------

def _write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def run_v1_analysis(
    results_dir: Union[Path, str],
    output_dir: Union[Path, str],
    settings: AnalysisSettings = AnalysisSettings(),
) -> Dict[str, pd.DataFrame]:
    """Create all current Sections 0–6 tables and figures from the final SQLite store.

    The source database is opened read-only. Output can contain patient-level aggregated
    predictions and must remain in secure storage; it is intentionally outside Git.
    """
    results = load_r1_results(results_dir)
    output_dir = Path(output_dir)
    tables_dir, figures_dir = output_dir / "tables", output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {
        "integrity": integrity_table(results),
        "preprocessing": preprocessing_table(),
        "phase_audit": phase_audit_table(results.phase_parameters),
        "runtime": runtime_table(results.run_metrics),
        "seed_stability_long": seed_stability_long(results.run_metrics),
        "seed_stability_main": seed_stability_main_table(results.run_metrics),
        "seed_confusion_long": seed_confusion_long(results.run_metrics),
        "seed_confusion_main": seed_confusion_table(results.run_metrics),
    }
    aggregated = aggregate_patient_probabilities(results.test_predictions, settings)
    tables["patient_aggregated_predictions"] = aggregated
    tables["patient_metrics_ci"] = patient_metric_table(aggregated, settings)
    tables["patient_confusion"] = final_confusion_table(aggregated, settings)
    tables["calibration"] = calibration_summary(aggregated, settings)
    tables["decision_curve"] = decision_curve_data(aggregated, settings)

    for name, table in tables.items():
        _write_table(table, tables_dir / f"{name}.csv")

    plot_stability_boxplots(results.run_metrics, figures_dir / "seed_stability_boxplots.png")
    plot_sensitivity_specificity(results.run_metrics, figures_dir / "sensitivity_specificity_tradeoff.png")
    plot_confusion_tradeoff(results.run_metrics, figures_dir / "false_positive_negative_tradeoff.png")
    plot_runtime_by_model(results.run_metrics, figures_dir / "runtime_by_model.png")
    plot_phase_epochs(results.phase_parameters, figures_dir / "phase_epochs_by_model.png")
    plot_roc(aggregated, figures_dir / "patient_level_roc.png")
    plot_calibration(aggregated, settings, figures_dir / "patient_level_calibration.png")
    plot_dca_overview(tables["decision_curve"], figures_dir / "patient_level_dca_overview.png")
    plot_dca_uncertainty(tables["decision_curve"], figures_dir / "patient_level_dca_uncertainty.png")

    manifest = {
        "analysis": "R1 TL results Version 1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_database": str(Path(results_dir) / "r1_final_tl_runs.sqlite"),
        "source_database_opened_read_only": True,
        "settings": asdict(settings),
        "scope": "Sections 0–6: TL only; excludes conventional baselines, CNN/reader comparison, and ensemble analysis.",
        "seed_level_statement": "100 runs are descriptive stability reporting only; no run-level p-values or best-seed selection are generated.",
        "patient_level_statement": "One final probability per patient/model is the arithmetic mean of 100 prespecified seed probabilities; binary classifications use the fixed 0.50 threshold.",
        "integrity": {"completed_runs": int(len(results.run_metrics)), "models": int(results.run_metrics.model_name.nunique()), "seeds_per_model": 100, "test_patients": 46},
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return tables
