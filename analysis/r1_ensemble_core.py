"""Safe development-only selection and historic-split ensemble analysis utilities.

The selection functions operate exclusively on one out-of-fold prediction per
patient and architecture from the 92-patient development cohort. The final-analysis
functions apply one already selected configuration to completed historic 61/31/46
transfer-learning predictions. This module never selects a configuration from test data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

RULE_ORDER: Tuple[str, ...] = ("sum", "median", "max", "majority_vote", "weighted_sum")
RULE_LABELS: Mapping[str, str] = {
    "sum": "Sum Rule",
    "median": "Median Rule",
    "max": "Max Rule",
    "majority_vote": "Majority Voting",
    "weighted_sum": "Weighted Sum Rule",
}
POOL_ORDER: Tuple[str, ...] = ("Top-3", "Top-5")
REQUIRED_OOF_COLUMNS = {"patient_id", "fold", "model_name", "observed_label", "probability"}


def file_sha256(path: Path) -> str:
    """Return a source file digest without copying a secure input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON record atomically to secure output storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Not JSON serializable: {0}".format(type(value).__name__))


def _binary_metrics(labels: Sequence[int], scores: Sequence[float], binary_predictions: Sequence[int]) -> Dict[str, Any]:
    labels_array = np.asarray(labels, dtype=int).reshape(-1)
    scores_array = np.asarray(scores, dtype=float).reshape(-1)
    predictions_array = np.asarray(binary_predictions, dtype=int).reshape(-1)
    if not (len(labels_array) == len(scores_array) == len(predictions_array)):
        raise ValueError("Labels, scores and predictions must have the same length")
    if not np.isin(labels_array, [0, 1]).all() or len(np.unique(labels_array)) != 2:
        raise ValueError("Both binary outcome classes are required")
    if not np.isfinite(scores_array).all() or not np.logical_and(scores_array >= 0.0, scores_array <= 1.0).all():
        raise ValueError("Scores must be finite values in [0, 1]")
    if not np.isin(predictions_array, [0, 1]).all():
        raise ValueError("Binary predictions must be 0/1")
    tn, fp, fn, tp = confusion_matrix(labels_array, predictions_array, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
        "precision": float(precision_score(labels_array, predictions_array, zero_division=0)),
        "sensitivity": float(recall_score(labels_array, predictions_array, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(labels_array, predictions_array, zero_division=0)),
        "auc": float(roc_auc_score(labels_array, scores_array)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# -----------------------------------------------------------------------------
# Development-only architecture/configuration selection
# -----------------------------------------------------------------------------

def validate_oof_predictions(frame: pd.DataFrame, expected_models: Sequence[str] = ()) -> pd.DataFrame:
    """Validate one OOF probability per development patient/architecture."""
    missing = sorted(REQUIRED_OOF_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("OOF predictions lack required columns: {0}".format(missing))
    result = frame.copy()
    result["patient_id"] = result["patient_id"].astype(str)
    result["model_name"] = result["model_name"].astype(str)
    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
    result["observed_label"] = pd.to_numeric(result["observed_label"], errors="raise").astype(int)
    result["probability"] = pd.to_numeric(result["probability"], errors="raise").astype(float)
    if result.empty or not np.isin(result["observed_label"], [0, 1]).all():
        raise ValueError("OOF predictions must contain binary labels")
    if not result["probability"].between(0.0, 1.0).all():
        raise ValueError("OOF probabilities must be finite values in [0, 1]")
    if result.duplicated(["patient_id", "model_name"]).any():
        raise ValueError("Every development patient requires exactly one OOF prediction per architecture")
    if result.groupby("patient_id")["observed_label"].nunique().ne(1).any():
        raise ValueError("Observed labels differ across architectures for a development patient")
    if result.groupby("patient_id")["fold"].nunique().ne(1).any():
        raise ValueError("Fold assignments differ across architectures for a development patient")
    model_patient_sets = result.groupby("model_name")["patient_id"].agg(lambda values: tuple(sorted(values)))
    if model_patient_sets.nunique() != 1:
        raise ValueError("All architectures must have OOF predictions for the same patients")
    if expected_models and set(result["model_name"]) != set(expected_models):
        raise ValueError("OOF architectures differ from the expected model set")
    return result.sort_values(["model_name", "patient_id"]).reset_index(drop=True)


def architecture_oof_ranking(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    """Rank architectures by pooled development OOF AUC."""
    oof = validate_oof_predictions(oof_predictions)
    rows: List[Dict[str, Any]] = []
    for model_name, group in oof.groupby("model_name", sort=True):
        values = group.sort_values("patient_id")
        binary = (values["probability"].to_numpy() >= 0.50).astype(int)
        metrics = _binary_metrics(values["observed_label"], values["probability"], binary)
        rows.append({
            "model_name": model_name,
            "n_oof_patients": int(len(values)),
            "n_folds": int(values["fold"].nunique()),
            "pooled_oof_auc": metrics["auc"],
            "pooled_oof_accuracy": metrics["accuracy"],
            "pooled_oof_precision": metrics["precision"],
            "pooled_oof_sensitivity": metrics["sensitivity"],
            "pooled_oof_specificity": metrics["specificity"],
            "pooled_oof_f1": metrics["f1"],
            "pooled_oof_tn": metrics["tn"], "pooled_oof_fp": metrics["fp"],
            "pooled_oof_fn": metrics["fn"], "pooled_oof_tp": metrics["tp"],
        })
    ranking = pd.DataFrame(rows).sort_values(["pooled_oof_auc", "model_name"], ascending=[False, True]).reset_index(drop=True)
    ranking.insert(0, "development_rank", np.arange(1, len(ranking) + 1, dtype=int))
    return ranking


def candidate_pools(architecture_ranking: pd.DataFrame) -> Dict[str, List[str]]:
    """Create the prespecified nested Top-3 and Top-5 pools."""
    if not {"development_rank", "model_name"}.issubset(architecture_ranking.columns):
        raise ValueError("Architecture ranking must include development_rank and model_name")
    ordered = architecture_ranking.sort_values("development_rank")["model_name"].astype(str).tolist()
    if len(ordered) < 5 or len(set(ordered)) != len(ordered):
        raise ValueError("At least five uniquely ranked architectures are required")
    return {"Top-3": ordered[:3], "Top-5": ordered[:5]}


def _normalized_oof_auc_weights(oof_predictions: pd.DataFrame, model_names: Sequence[str]) -> Dict[str, float]:
    subset = oof_predictions.loc[oof_predictions["model_name"].isin(model_names)]
    ranking = architecture_oof_ranking(subset).set_index("model_name")
    auc_values = ranking.loc[list(model_names), "pooled_oof_auc"].to_numpy(dtype=float)
    if not np.isfinite(auc_values).all() or np.any(auc_values <= 0.0):
        raise ValueError("Development OOF AUC weights must be positive finite values")
    normalized = auc_values / auc_values.sum()
    return {name: float(weight) for name, weight in zip(model_names, normalized)}


def _apply_rule(probability_matrix: np.ndarray, rule_id: str, weights: Sequence[float] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one allowed rule to [n_constituents, n_patients] probabilities."""
    probabilities = np.asarray(probability_matrix, dtype=float)
    if probabilities.ndim != 2 or min(probabilities.shape) < 1:
        raise ValueError("Probability matrix must have shape [n_models, n_patients]")
    if not np.isfinite(probabilities).all() or not np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all():
        raise ValueError("Constituent probabilities must be finite values in [0, 1]")
    if rule_id == "sum":
        scores = probabilities.mean(axis=0)
        binary = (scores >= 0.50).astype(int)
    elif rule_id == "median":
        scores = np.median(probabilities, axis=0)
        binary = (scores >= 0.50).astype(int)
    elif rule_id == "max":
        scores = probabilities.max(axis=0)
        binary = (scores >= 0.50).astype(int)
    elif rule_id == "majority_vote":
        scores = (probabilities >= 0.50).mean(axis=0)
        binary = (scores > 0.50).astype(int)
    elif rule_id == "weighted_sum":
        weight_array = np.asarray(weights, dtype=float).reshape(-1)
        if len(weight_array) != probabilities.shape[0] or not np.isfinite(weight_array).all() or np.any(weight_array <= 0.0):
            raise ValueError("Weighted Sum requires one positive finite weight for each constituent")
        scores = np.dot(weight_array / weight_array.sum(), probabilities)
        binary = (scores >= 0.50).astype(int)
    else:
        raise ValueError("Unsupported ensemble rule: {0}".format(rule_id))
    return scores.astype(float), binary.astype(int)


def configuration_predictions(oof_predictions: pd.DataFrame, pool_name: str, model_names: Sequence[str], rule_id: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Create OOF predictions for one development-only candidate configuration."""
    if pool_name not in POOL_ORDER or rule_id not in RULE_ORDER:
        raise ValueError("Unsupported pool or ensemble rule")
    oof = validate_oof_predictions(oof_predictions)
    models = [str(name) for name in model_names]
    if len(models) != len(set(models)) or not set(models).issubset(set(oof["model_name"])):
        raise ValueError("Candidate pool contains invalid architectures")
    subset = oof.loc[oof["model_name"].isin(models)]
    patient_table = subset[["patient_id", "fold", "observed_label"]].drop_duplicates().sort_values("patient_id").reset_index(drop=True)
    if len(subset) != len(patient_table) * len(models):
        raise ValueError("Candidate configuration has incomplete OOF predictions")
    wide = subset.pivot(index="patient_id", columns="model_name", values="probability").loc[patient_table["patient_id"], models]
    weights_by_model: Dict[str, float] = {}
    if rule_id == "weighted_sum":
        weights_by_model = _normalized_oof_auc_weights(oof, models)
        weights = [weights_by_model[name] for name in models]
    else:
        weights = []
    scores, binary = _apply_rule(wide.to_numpy(dtype=float).T, rule_id, weights)
    result = patient_table.copy()
    result["pool_name"] = pool_name
    result["rule_id"] = rule_id
    result["rule_name"] = RULE_LABELS[rule_id]
    result["configuration_id"] = "{0}_{1}".format(pool_name.lower().replace("-", ""), rule_id)
    result["constituent_models"] = ";".join(models)
    result["score"] = scores
    result["binary_prediction"] = binary
    return result, weights_by_model


def configuration_ranking(oof_predictions: pd.DataFrame, pools: Mapping[str, Sequence[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Evaluate exactly ten Top-3/Top-5 × five-rule development configurations."""
    oof = validate_oof_predictions(oof_predictions)
    if tuple(pools.keys()) != POOL_ORDER:
        raise ValueError("Candidate pools must be ordered Top-3 then Top-5")
    ranking_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    weights_by_configuration: Dict[str, Dict[str, float]] = {}
    for pool_priority, pool_name in enumerate(POOL_ORDER, start=1):
        models = list(pools[pool_name])
        expected_size = 3 if pool_name == "Top-3" else 5
        if len(models) != expected_size:
            raise ValueError("{0} must contain exactly {1} models".format(pool_name, expected_size))
        for rule_priority, rule_id in enumerate(RULE_ORDER, start=1):
            predictions, weights = configuration_predictions(oof, pool_name, models, rule_id)
            metrics = _binary_metrics(predictions["observed_label"], predictions["score"], predictions["binary_prediction"])
            configuration_id = str(predictions["configuration_id"].iloc[0])
            weights_by_configuration[configuration_id] = weights
            prediction_frames.append(predictions)
            ranking_rows.append({
                "configuration_id": configuration_id,
                "pool_name": pool_name,
                "n_constituent_models": len(models),
                "rule_id": rule_id,
                "rule_name": RULE_LABELS[rule_id],
                "constituent_models": ";".join(models),
                "weights_source": "pooled development OOF AUC" if rule_id == "weighted_sum" else "not applicable",
                "weights": json.dumps(weights, sort_keys=True) if weights else "",
                "n_oof_patients": int(len(predictions)),
                "pooled_oof_auc": metrics["auc"],
                "pooled_oof_accuracy": metrics["accuracy"],
                "pooled_oof_precision": metrics["precision"],
                "pooled_oof_sensitivity": metrics["sensitivity"],
                "pooled_oof_specificity": metrics["specificity"],
                "pooled_oof_f1": metrics["f1"],
                "pooled_oof_tn": metrics["tn"], "pooled_oof_fp": metrics["fp"],
                "pooled_oof_fn": metrics["fn"], "pooled_oof_tp": metrics["tp"],
                "_pool_priority": pool_priority,
                "_rule_priority": rule_priority,
            })
    ranking = pd.DataFrame(ranking_rows)
    if len(ranking) != 10:
        raise RuntimeError("Expected ten candidate configurations; found {0}".format(len(ranking)))
    ranking["selection_auc_rounded_12dp"] = ranking["pooled_oof_auc"].round(12)
    ranking = ranking.sort_values(
        ["selection_auc_rounded_12dp", "_pool_priority", "_rule_priority", "configuration_id"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    ranking.insert(0, "development_rank", np.arange(1, len(ranking) + 1, dtype=int))
    ranking = ranking.drop(columns=["_pool_priority", "_rule_priority"])
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["configuration_id", "patient_id"]).reset_index(drop=True)
    return ranking, predictions, weights_by_configuration


def selection_lock(architecture_ranking_frame: pd.DataFrame, configuration_ranking_frame: pd.DataFrame, pools: Mapping[str, Sequence[str]], weights_by_configuration: Mapping[str, Mapping[str, float]], oof_file: Path, cv_manifest_file: Path) -> Dict[str, Any]:
    """Create a durable rank-1 development-only selection record."""
    if configuration_ranking_frame.empty:
        raise ValueError("Configuration ranking is empty")
    selected = configuration_ranking_frame.sort_values("development_rank").iloc[0]
    configuration_id = str(selected["configuration_id"])
    selected_pool = str(selected["pool_name"])
    return {
        "purpose": "development_only_ensemble_configuration_selection",
        "test_data_accessed": False,
        "selection_dataset": "92-patient development cohort only",
        "selection_metric": "highest pooled out-of-fold AUC",
        "tie_breakers": [
            "exact numerical AUC tie at 12 decimal places: smaller constituent pool (Top-3 before Top-5)",
            "remaining exact tie: Sum, Median, Max, Majority Voting, Weighted Sum",
        ],
        "candidate_pools": {name: list(values) for name, values in pools.items()},
        "candidate_rules": [{"rule_id": rule, "rule_name": RULE_LABELS[rule]} for rule in RULE_ORDER],
        "n_candidate_configurations": 10,
        "oof_predictions_file": str(oof_file),
        "oof_predictions_sha256": file_sha256(oof_file),
        "cv_run_manifest_file": str(cv_manifest_file),
        "cv_run_manifest_sha256": file_sha256(cv_manifest_file),
        "architecture_ranking": architecture_ranking_frame.to_dict(orient="records"),
        "selected_configuration": {
            "configuration_id": configuration_id,
            "pool_name": selected_pool,
            "rule_id": str(selected["rule_id"]),
            "rule_name": str(selected["rule_name"]),
            "constituent_models": list(pools[selected_pool]),
            "classification_threshold": 0.50,
            "weighted_sum_weights": dict(weights_by_configuration.get(configuration_id, {})),
            "selection_result": {
                "pooled_oof_auc": float(selected["pooled_oof_auc"]),
                "pooled_oof_accuracy": float(selected["pooled_oof_accuracy"]),
                "pooled_oof_sensitivity": float(selected["pooled_oof_sensitivity"]),
                "pooled_oof_specificity": float(selected["pooled_oof_specificity"]),
                "pooled_oof_f1": float(selected["pooled_oof_f1"]),
            },
        },
        "final_test_evaluation_rule": "Apply only this selected configuration to run-matched predictions from the historic 61/31/46 TL benchmark.",
    }


# -----------------------------------------------------------------------------
# Final analysis of one manually entered, development-selected configuration
# -----------------------------------------------------------------------------

def _parse_probability_csv(value: Any, expected_length: int) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        probabilities = np.asarray(value, dtype=float).reshape(-1)
    else:
        text = str(value).strip().strip("[](){}")
        probabilities = np.asarray([float(item.strip()) for item in text.split(",") if item.strip()], dtype=float)
    if len(probabilities) != expected_length:
        raise ValueError("Expected {0} probabilities; found {1}".format(expected_length, len(probabilities)))
    if not np.isfinite(probabilities).all() or not np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all():
        raise ValueError("Probability values must be finite values in [0, 1]")
    return probabilities


def load_historic_tl_workbook(workbook_path: Path, expected_models: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the final Experiment2026 workbook and validate 11×100 historic predictions."""
    if not workbook_path.exists():
        raise FileNotFoundError("Historic TL workbook not found: {0}".format(workbook_path))
    sheets = pd.read_excel(str(workbook_path), sheet_name=None, engine="openpyxl")
    missing_sheets = sorted({"Run_results", "Patient_manifest", "Study_lock"} - set(sheets))
    if missing_sheets:
        raise ValueError("Historic TL workbook lacks required sheets: {0}".format(missing_sheets))
    runs = sheets["Run_results"].copy()
    manifest = sheets["Patient_manifest"].copy()
    study_lock = sheets["Study_lock"].copy()
    missing_columns = sorted({"model_name", "seed", "test_probabilities"} - set(runs.columns))
    if missing_columns:
        raise ValueError("Run_results lacks required columns: {0}".format(missing_columns))
    required_manifest = {"split", "within_split_order", "file_name", "observed_label"}
    if not required_manifest.issubset(manifest.columns):
        raise ValueError("Patient_manifest lacks required columns")
    if set(runs["model_name"].astype(str)) != set(expected_models) or len(runs) != len(expected_models) * 100:
        raise ValueError("Historic TL workbook must contain exactly 11 architectures × 100 runs")
    if runs.duplicated(["model_name", "seed"]).any():
        raise ValueError("Historic TL workbook contains duplicate model/seed rows")
    for model_name in expected_models:
        seeds = set(pd.to_numeric(runs.loc[runs["model_name"].astype(str) == model_name, "seed"], errors="raise").astype(int))
        if seeds != set(range(1, 101)):
            raise ValueError("Model {0} must have exactly seeds 1-100".format(model_name))
    test_manifest = manifest.loc[manifest["split"].astype(str) == "test"].copy()
    test_manifest["within_split_order"] = pd.to_numeric(test_manifest["within_split_order"], errors="raise").astype(int)
    test_manifest["observed_label"] = pd.to_numeric(test_manifest["observed_label"], errors="raise").astype(int)
    test_manifest = test_manifest.sort_values("within_split_order").reset_index(drop=True)
    if len(test_manifest) != 46 or test_manifest["within_split_order"].tolist() != list(range(1, 47)):
        raise ValueError("Patient_manifest must contain exactly 46 ordered test patients")
    if (int((test_manifest["observed_label"] == 0).sum()), int((test_manifest["observed_label"] == 1).sum())) != (26, 20):
        raise ValueError("Historic test labels must have class 0=26 and class 1=20")
    rows: List[Dict[str, Any]] = []
    for row in runs.itertuples(index=False):
        values = row._asdict()
        probabilities = _parse_probability_csv(values["test_probabilities"], 46)
        for order, patient_id, label, probability in zip(test_manifest["within_split_order"], test_manifest["file_name"], test_manifest["observed_label"], probabilities):
            rows.append({
                "patient_id": str(patient_id), "within_split_order": int(order), "observed_label": int(label),
                "model_name": str(values["model_name"]), "seed": int(values["seed"]), "probability": float(probability),
                "binary_prediction": int(float(probability) >= 0.50),
            })
    predictions = pd.DataFrame(rows).sort_values(["model_name", "seed", "within_split_order"]).reset_index(drop=True)
    return predictions, test_manifest, study_lock


def validate_manual_configuration(top3_models: Sequence[str], top5_models: Sequence[str], selected_pool: str, selected_rule: str, weighted_sum_weights: Mapping[str, float] = None) -> Dict[str, Any]:
    """Validate the one user-entered recipe copied from development selection."""
    top3 = [str(name) for name in top3_models]
    top5 = [str(name) for name in top5_models]
    if len(top3) != 3 or len(top5) != 5 or len(set(top5)) != 5 or top3 != top5[:3]:
        raise ValueError("Enter nested Top-3 and Top-5 development-ranked model lists")
    if selected_pool not in POOL_ORDER or selected_rule not in RULE_ORDER:
        raise ValueError("Unsupported selected pool or rule")
    selected_models = top3 if selected_pool == "Top-3" else top5
    weights = {str(name): float(value) for name, value in (weighted_sum_weights or {}).items()}
    if selected_rule == "weighted_sum":
        if set(weights) != set(selected_models) or not np.isfinite(np.asarray(list(weights.values()), dtype=float)).all() or any(value <= 0.0 for value in weights.values()):
            raise ValueError("Weighted Sum requires one positive development-derived weight per selected model")
        total = sum(weights.values())
        weights = {name: float(value / total) for name, value in weights.items()}
    elif weights:
        raise ValueError("Only the Weighted Sum Rule uses weights")
    return {
        "top3_models": top3, "top5_models": top5,
        "selected_pool": selected_pool, "selected_rule": selected_rule,
        "selected_rule_name": RULE_LABELS[selected_rule],
        "selected_models": selected_models,
        "weighted_sum_weights": weights,
        "classification_threshold": 0.50,
        "test_configuration_selection_performed": False,
    }


def create_run_matched_ensemble_predictions(historic_predictions: pd.DataFrame, configuration: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create exactly 100 ensemble replicates from same-seed constituent predictions."""
    required = {"patient_id", "observed_label", "model_name", "seed", "probability"}
    if not required.issubset(historic_predictions.columns):
        raise ValueError("Historic prediction table lacks required columns")
    models = [str(name) for name in configuration["selected_models"]]
    rule_id = str(configuration["selected_rule"])
    subset = historic_predictions.loc[historic_predictions["model_name"].astype(str).isin(models)].copy()
    if len(subset) != len(models) * 100 * 46 or subset.duplicated(["patient_id", "model_name", "seed"]).any():
        raise ValueError("Historic predictions are incomplete for the selected constituent models")
    if "within_split_order" in subset.columns:
        patient_table = subset[["patient_id", "observed_label", "within_split_order"]].drop_duplicates().sort_values("within_split_order").reset_index(drop=True)
    else:
        patient_table = subset[["patient_id", "observed_label"]].drop_duplicates().sort_values("patient_id").reset_index(drop=True)
    if len(patient_table) != 46:
        raise ValueError("Expected exactly 46 independent test patients")
    prediction_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    weights_by_model = dict(configuration.get("weighted_sum_weights", {}))
    for seed in range(1, 101):
        seed_frame = subset.loc[subset["seed"].astype(int) == seed]
        wide = seed_frame.pivot(index="patient_id", columns="model_name", values="probability")
        if len(wide) != 46 or set(wide.columns.astype(str)) != set(models):
            raise ValueError("Seed {0} lacks complete run-matched predictions".format(seed))
        wide = wide.loc[patient_table["patient_id"], models]
        weights = [weights_by_model[name] for name in models] if rule_id == "weighted_sum" else []
        scores, binary = _apply_rule(wide.to_numpy(dtype=float).T, rule_id, weights)
        metrics = _binary_metrics(patient_table["observed_label"], scores, binary)
        metrics_rows.append({"method": "Ensemble", "seed": seed, "pool_name": configuration["selected_pool"], "rule_id": rule_id, "rule_name": configuration["selected_rule_name"], **metrics})
        for index, (patient_id, label, score, prediction) in enumerate(zip(patient_table["patient_id"], patient_table["observed_label"], scores, binary), start=1):
            prediction_rows.append({"method": "Ensemble", "patient_id": patient_id, "within_split_order": index, "observed_label": int(label), "seed": seed, "pool_name": configuration["selected_pool"], "rule_id": rule_id, "rule_name": configuration["selected_rule_name"], "probability": float(score), "binary_prediction": int(prediction)})
    return pd.DataFrame(prediction_rows), pd.DataFrame(metrics_rows)


def run_stability_summary(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive median/IQR stability across exactly 100 replicates."""
    required = {"seed", "accuracy", "precision", "sensitivity", "specificity", "f1", "auc"}
    if not required.issubset(run_metrics.columns) or set(pd.to_numeric(run_metrics["seed"], errors="raise").astype(int)) != set(range(1, 101)):
        raise ValueError("Stability summary requires seeds 1-100 and all six metrics")
    row: Dict[str, Any] = {"method": str(run_metrics["method"].iloc[0]) if "method" in run_metrics else "Ensemble", "n_runs": int(len(run_metrics))}
    for input_name, output_name in (("accuracy", "ACC"), ("precision", "PRE"), ("sensitivity", "SEN"), ("specificity", "SPE"), ("f1", "F1S"), ("auc", "AUC")):
        values = pd.to_numeric(run_metrics[input_name], errors="raise").to_numpy(dtype=float)
        row["{0}_median".format(output_name)] = float(np.median(values))
        row["{0}_iqr".format(output_name)] = float(np.percentile(values, 75) - np.percentile(values, 25))
    return pd.DataFrame([row])


def raw_ensemble_run_results(
    ensemble_predictions: pd.DataFrame,
    ensemble_run_metrics: pd.DataFrame,
    configuration: Mapping[str, Any],
) -> pd.DataFrame:
    """Create one raw run row per ensemble seed for later combined reporting.

    This deliberately does not calculate medians, confidence intervals, or tests.
    It mirrors the run-level structure of the historic TL workbook: each row holds
    one seed's metrics and its full-precision ordered 46-patient probabilities.
    """
    expected_prediction_columns = {"seed", "within_split_order", "probability", "binary_prediction"}
    expected_metric_columns = {"seed", "accuracy", "precision", "sensitivity", "specificity", "f1", "auc", "tn", "fp", "fn", "tp"}
    if not expected_prediction_columns.issubset(ensemble_predictions.columns):
        raise ValueError("Ensemble predictions lack columns required for raw workbook output")
    if not expected_metric_columns.issubset(ensemble_run_metrics.columns):
        raise ValueError("Ensemble run metrics lack columns required for raw workbook output")
    rows: List[Dict[str, Any]] = []
    selected_models = ";".join(str(name) for name in configuration["selected_models"])
    for metrics in ensemble_run_metrics.sort_values("seed").itertuples(index=False):
        values = metrics._asdict()
        seed = int(values["seed"])
        predictions = ensemble_predictions.loc[ensemble_predictions["seed"].astype(int) == seed].sort_values("within_split_order")
        if len(predictions) != 46 or predictions["within_split_order"].tolist() != list(range(1, 47)):
            raise ValueError("Each raw ensemble run must contain 46 ordered test predictions")
        probabilities = predictions["probability"].to_numpy(dtype=float)
        binary = predictions["binary_prediction"].to_numpy(dtype=int)
        rows.append({
            "model_name": "Ensemble",
            "seed": seed,
            "pool_name": str(configuration["selected_pool"]),
            "rule_id": str(configuration["selected_rule"]),
            "rule_name": str(configuration["selected_rule_name"]),
            "constituent_models": selected_models,
            "weighted_sum_weights": json.dumps(configuration.get("weighted_sum_weights", {}), sort_keys=True),
            "threshold": 0.50,
            "test_accuracy": float(values["accuracy"]),
            "test_precision": float(values["precision"]),
            "test_sensitivity": float(values["sensitivity"]),
            "test_specificity": float(values["specificity"]),
            "test_f1": float(values["f1"]),
            "test_auc": float(values["auc"]),
            "test_tn": int(values["tn"]),
            "test_fp": int(values["fp"]),
            "test_fn": int(values["fn"]),
            "test_tp": int(values["tp"]),
            "test_probabilities": ",".join(format(float(value), ".17g") for value in probabilities),
            "test_binary_predictions": ",".join(str(int(value)) for value in binary),
        })
    result = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    if len(result) != 100 or set(result["seed"]) != set(range(1, 101)):
        raise RuntimeError("Raw ensemble workbook must contain exactly one row for seeds 1-100")
    return result


def raw_ensemble_study_lock(configuration: Mapping[str, Any], input_workbook: Path) -> pd.DataFrame:
    """Create the compact provenance sheet for the raw ensemble workbook."""
    rows = [
        ("study", "EJPH-D-26-00179 final ensemble raw 100-run output"),
        ("input_workbook", str(input_workbook)),
        ("input_design", "Completed historic 61/31/46 transfer-learning workbook: 11 architectures x 100 seeds"),
        ("test_configuration_selection_performed", "false"),
        ("selected_pool", str(configuration["selected_pool"])),
        ("selected_rule", str(configuration["selected_rule"])),
        ("constituent_models", ";".join(str(name) for name in configuration["selected_models"])),
        ("weighted_sum_weights", json.dumps(configuration.get("weighted_sum_weights", {}), sort_keys=True)),
        ("threshold", "0.50"),
        ("seed_matching", "same seed across every selected constituent model"),
        ("n_ensemble_runs", "100"),
        ("n_test_patients_per_run", "46"),
        ("output_scope", "Raw 100-run ensemble results only; later combined analysis performs stability and patient-level reporting"),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def patient_level_summary(ensemble_predictions: pd.DataFrame, bootstrap_iterations: int = 2000, random_seed: int = 20260929) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mean 100 replicates per patient and calculate patient-level bootstrap CIs."""
    required = {"patient_id", "observed_label", "seed", "probability"}
    if not required.issubset(ensemble_predictions.columns):
        raise ValueError("Ensemble predictions lack required columns")
    seed_counts = ensemble_predictions.groupby("patient_id")["seed"].nunique()
    if len(seed_counts) != 46 or not seed_counts.eq(100).all():
        raise ValueError("Each of 46 test patients must have exactly 100 ensemble predictions")
    patient_columns = ["patient_id", "observed_label"]
    if "within_split_order" in ensemble_predictions.columns:
        patient_columns.append("within_split_order")
    grouped = ensemble_predictions.groupby(patient_columns, as_index=False)["probability"].mean().rename(columns={"probability": "mean_probability"})
    grouped = grouped.sort_values("within_split_order" if "within_split_order" in grouped else "patient_id").reset_index(drop=True)
    grouped["final_binary_prediction"] = (grouped["mean_probability"] >= 0.50).astype(int)
    point_metrics = _binary_metrics(grouped["observed_label"], grouped["mean_probability"], grouped["final_binary_prediction"])
    labels = grouped["observed_label"].to_numpy(dtype=int)
    probabilities = grouped["mean_probability"].to_numpy(dtype=float)
    predictions = grouped["final_binary_prediction"].to_numpy(dtype=int)
    rng = np.random.RandomState(random_seed)
    metric_names = ("accuracy", "precision", "sensitivity", "specificity", "f1", "auc")
    samples: MutableMapping[str, List[float]] = {name: [] for name in metric_names}
    attempts = 0
    while len(samples["auc"]) < bootstrap_iterations and attempts < bootstrap_iterations * 5 + 100:
        attempts += 1
        indices = rng.randint(0, len(grouped), size=len(grouped))
        if len(np.unique(labels[indices])) != 2:
            continue
        bootstrap = _binary_metrics(labels[indices], probabilities[indices], predictions[indices])
        for name in metric_names:
            samples[name].append(float(bootstrap[name]))
    if len(samples["auc"]) != bootstrap_iterations:
        raise RuntimeError("Insufficient valid bootstrap samples")
    result: Dict[str, Any] = {"method": "Ensemble", "n_test_patients": int(len(grouped)), "bootstrap_iterations": int(bootstrap_iterations)}
    for name in metric_names:
        result["{0}_point".format(name)] = float(point_metrics[name])
        result["{0}_ci_low".format(name)] = float(np.percentile(samples[name], 2.5))
        result["{0}_ci_high".format(name)] = float(np.percentile(samples[name], 97.5))
    for name in ("tn", "fp", "fn", "tp"):
        result[name] = int(point_metrics[name])
    return grouped, pd.DataFrame([result])
