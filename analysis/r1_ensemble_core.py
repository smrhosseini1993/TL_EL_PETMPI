"""Development-only ensemble-selection utilities for the R1 PET-MPI revision.

This module contains no patient data and never loads the independent test cohort.
It combines one out-of-fold probability per development patient and architecture to
rank the predeclared Top-3/Top-5 ensemble configurations. The same selected rule
can subsequently be applied to run-matched historic-split test predictions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


RULE_ORDER: Tuple[str, ...] = (
    "sum",
    "median",
    "max",
    "majority_vote",
    "weighted_sum",
)
RULE_LABELS: Mapping[str, str] = {
    "sum": "Sum Rule",
    "median": "Median Rule",
    "max": "Max Rule",
    "majority_vote": "Majority Voting",
    "weighted_sum": "Weighted Sum Rule",
}
POOL_ORDER: Tuple[str, ...] = ("Top-3", "Top-5")
REQUIRED_OOF_COLUMNS = {
    "patient_id",
    "fold",
    "model_name",
    "observed_label",
    "probability",
}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a secure input file without copying it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _binary_metrics(labels: Sequence[int], scores: Sequence[float], binary_predictions: Sequence[int]) -> Dict[str, Any]:
    y_true = np.asarray(labels, dtype=int).reshape(-1)
    y_score = np.asarray(scores, dtype=float).reshape(-1)
    y_pred = np.asarray(binary_predictions, dtype=int).reshape(-1)
    if not (len(y_true) == len(y_score) == len(y_pred)):
        raise ValueError("Labels, scores and binary predictions must have equal length")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Observed labels must be binary 0/1")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Both outcome classes are required to calculate AUC")
    if not np.isfinite(y_score).all() or not np.logical_and(y_score >= 0.0, y_score <= 1.0).all():
        raise ValueError("Ensemble scores must be finite values in [0, 1]")
    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Binary ensemble predictions must be 0/1")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def validate_oof_predictions(frame: pd.DataFrame, expected_models: Sequence[str] = ()) -> pd.DataFrame:
    """Validate one OOF probability per patient/model and return a sorted copy."""
    missing = sorted(REQUIRED_OOF_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("OOF prediction table is missing required columns: {0}".format(missing))
    result = frame.copy()
    result["patient_id"] = result["patient_id"].astype(str)
    result["model_name"] = result["model_name"].astype(str)
    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
    result["observed_label"] = pd.to_numeric(result["observed_label"], errors="raise").astype(int)
    result["probability"] = pd.to_numeric(result["probability"], errors="raise").astype(float)
    if result.empty:
        raise ValueError("OOF prediction table is empty")
    if not np.isin(result["observed_label"], [0, 1]).all():
        raise ValueError("OOF observed_label values must be binary 0/1")
    if not result["probability"].between(0.0, 1.0).all():
        raise ValueError("OOF probabilities must be finite values in [0, 1]")
    if result.duplicated(["patient_id", "model_name"]).any():
        raise ValueError("Each development patient must have exactly one OOF prediction per architecture")
    if result.groupby("patient_id")["observed_label"].nunique().ne(1).any():
        raise ValueError("A development patient has inconsistent observed labels across architectures")
    if result.groupby("patient_id")["fold"].nunique().ne(1).any():
        raise ValueError("A development patient has inconsistent folds across architectures")
    patient_sets = result.groupby("model_name")["patient_id"].agg(lambda values: tuple(sorted(values)))
    if patient_sets.nunique() != 1:
        raise ValueError("Every architecture must cover the identical development-patient set")
    if expected_models:
        observed = set(result["model_name"])
        expected = set(expected_models)
        if observed != expected:
            raise ValueError("OOF models differ from the expected architecture set; expected={0}, observed={1}".format(sorted(expected), sorted(observed)))
    return result.sort_values(["model_name", "patient_id"]).reset_index(drop=True)


def architecture_oof_ranking(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    """Rank architectures by pooled development-only OOF AUC."""
    oof = validate_oof_predictions(oof_predictions)
    rows: List[Dict[str, Any]] = []
    for model_name, group in oof.groupby("model_name", sort=True):
        scores = group.sort_values("patient_id")
        binary = (scores["probability"].to_numpy() >= 0.50).astype(int)
        metrics = _binary_metrics(scores["observed_label"], scores["probability"], binary)
        rows.append({
            "model_name": model_name,
            "n_oof_patients": int(len(scores)),
            "n_folds": int(scores["fold"].nunique()),
            "pooled_oof_auc": metrics["auc"],
            "pooled_oof_accuracy": metrics["accuracy"],
            "pooled_oof_precision": metrics["precision"],
            "pooled_oof_sensitivity": metrics["sensitivity"],
            "pooled_oof_specificity": metrics["specificity"],
            "pooled_oof_f1": metrics["f1"],
            "pooled_oof_tn": metrics["tn"],
            "pooled_oof_fp": metrics["fp"],
            "pooled_oof_fn": metrics["fn"],
            "pooled_oof_tp": metrics["tp"],
        })
    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(["pooled_oof_auc", "model_name"], ascending=[False, True]).reset_index(drop=True)
    ranking.insert(0, "development_rank", np.arange(1, len(ranking) + 1, dtype=int))
    return ranking


def candidate_pools(architecture_ranking: pd.DataFrame) -> Dict[str, List[str]]:
    """Build the prespecified Top-3 and Top-5 pools from the OOF ranking."""
    required = {"development_rank", "model_name"}
    if not required.issubset(architecture_ranking.columns):
        raise ValueError("Architecture ranking must contain development_rank and model_name")
    ordered = architecture_ranking.sort_values("development_rank")["model_name"].astype(str).tolist()
    if len(ordered) < 5:
        raise ValueError("At least five architectures are required for Top-3 and Top-5 pools")
    if len(set(ordered)) != len(ordered):
        raise ValueError("Architecture ranking contains duplicate model names")
    return {"Top-3": ordered[:3], "Top-5": ordered[:5]}


def _normalized_weights(oof_predictions: pd.DataFrame, model_names: Sequence[str]) -> Dict[str, float]:
    """Calculate weights from pooled development OOF AUC only."""
    subset = oof_predictions.loc[oof_predictions["model_name"].isin(model_names)].copy()
    ranking = architecture_oof_ranking(subset).set_index("model_name")
    ordered_auc = ranking.loc[list(model_names), "pooled_oof_auc"].to_numpy(dtype=float)
    if not np.isfinite(ordered_auc).all() or np.any(ordered_auc <= 0.0):
        raise ValueError("Development OOF AUC weights must be positive finite values")
    weights = ordered_auc / ordered_auc.sum()
    return {name: float(weight) for name, weight in zip(model_names, weights)}


def _apply_rule(probability_matrix: np.ndarray, rule_id: str, weights: Sequence[float] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one prespecified combination rule to [n_models, n_patients] probabilities."""
    probabilities = np.asarray(probability_matrix, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] < 1 or probabilities.shape[1] < 1:
        raise ValueError("Probability matrix must have shape [n_models, n_patients]")
    if not np.isfinite(probabilities).all() or not np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all():
        raise ValueError("All constituent probabilities must be finite values in [0, 1]")
    if rule_id == "majority_vote":
        scores = (probabilities >= 0.50).mean(axis=0)
        predicted = (scores > 0.50).astype(int)
    elif rule_id == "sum":
        scores = probabilities.mean(axis=0)
        predicted = (scores >= 0.50).astype(int)
    elif rule_id == "weighted_sum":
        weight_array = np.asarray(weights, dtype=float).reshape(-1)
        if len(weight_array) != probabilities.shape[0] or not np.isfinite(weight_array).all() or np.any(weight_array <= 0.0):
            raise ValueError("Weighted Sum Rule requires one positive finite weight for each constituent model")
        normalized = weight_array / weight_array.sum()
        scores = np.dot(normalized, probabilities)
        predicted = (scores >= 0.50).astype(int)
    elif rule_id == "max":
        scores = probabilities.max(axis=0)
        predicted = (scores >= 0.50).astype(int)
    elif rule_id == "median":
        scores = np.median(probabilities, axis=0)
        predicted = (scores >= 0.50).astype(int)
    else:
        raise ValueError("Unsupported ensemble rule: {0}".format(rule_id))
    return scores.astype(float), predicted.astype(int)


def configuration_predictions(
    oof_predictions: pd.DataFrame,
    pool_name: str,
    model_names: Sequence[str],
    rule_id: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Create development OOF predictions for one candidate ensemble configuration."""
    if pool_name not in POOL_ORDER:
        raise ValueError("Unsupported pool name: {0}".format(pool_name))
    if rule_id not in RULE_ORDER:
        raise ValueError("Unsupported rule ID: {0}".format(rule_id))
    oof = validate_oof_predictions(oof_predictions)
    ordered_models = [str(model_name) for model_name in model_names]
    if len(ordered_models) != len(set(ordered_models)):
        raise ValueError("A candidate pool contains duplicate models")
    observed_models = set(oof["model_name"])
    missing_models = sorted(set(ordered_models) - observed_models)
    if missing_models:
        raise ValueError("Candidate pool models are absent from OOF predictions: {0}".format(missing_models))
    subset = oof.loc[oof["model_name"].isin(ordered_models)].copy()
    patient_table = (
        subset[["patient_id", "fold", "observed_label"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    if len(patient_table) * len(ordered_models) != len(subset):
        raise ValueError("Candidate configuration lacks one or more patient/model OOF probabilities")
    wide = subset.pivot(index="patient_id", columns="model_name", values="probability")
    wide = wide.loc[patient_table["patient_id"], ordered_models]
    weights_by_model: Dict[str, float] = {}
    if rule_id == "weighted_sum":
        weights_by_model = _normalized_weights(oof, ordered_models)
        weights = [weights_by_model[model_name] for model_name in ordered_models]
    else:
        weights = []
    scores, predicted = _apply_rule(wide.to_numpy(dtype=float).T, rule_id, weights)
    result = patient_table.copy()
    result["pool_name"] = pool_name
    result["rule_id"] = rule_id
    result["rule_name"] = RULE_LABELS[rule_id]
    result["configuration_id"] = "{0}_{1}".format(pool_name.lower().replace("-", ""), rule_id)
    result["constituent_models"] = ";".join(ordered_models)
    result["score"] = scores
    result["binary_prediction"] = predicted
    return result, weights_by_model


def configuration_ranking(oof_predictions: pd.DataFrame, pools: Mapping[str, Sequence[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Evaluate exactly ten development-only ensemble configurations and rank them."""
    oof = validate_oof_predictions(oof_predictions)
    if tuple(pools.keys()) != POOL_ORDER:
        raise ValueError("Candidate pools must be ordered exactly as Top-3 then Top-5")
    prediction_frames: List[pd.DataFrame] = []
    ranking_rows: List[Dict[str, Any]] = []
    all_weights: Dict[str, Dict[str, float]] = {}
    for pool_priority, pool_name in enumerate(POOL_ORDER, start=1):
        model_names = list(pools[pool_name])
        expected_size = 3 if pool_name == "Top-3" else 5
        if len(model_names) != expected_size:
            raise ValueError("{0} must contain exactly {1} models".format(pool_name, expected_size))
        for rule_priority, rule_id in enumerate(RULE_ORDER, start=1):
            predictions, weights = configuration_predictions(oof, pool_name, model_names, rule_id)
            metrics = _binary_metrics(predictions["observed_label"], predictions["score"], predictions["binary_prediction"])
            config_id = str(predictions["configuration_id"].iloc[0])
            prediction_frames.append(predictions)
            all_weights[config_id] = weights
            ranking_rows.append({
                "configuration_id": config_id,
                "pool_name": pool_name,
                "n_constituent_models": len(model_names),
                "rule_id": rule_id,
                "rule_name": RULE_LABELS[rule_id],
                "constituent_models": ";".join(model_names),
                "weights_source": "pooled development OOF AUC" if rule_id == "weighted_sum" else "not applicable",
                "weights": json.dumps(weights, sort_keys=True) if weights else "",
                "n_oof_patients": int(len(predictions)),
                "pooled_oof_auc": metrics["auc"],
                "pooled_oof_accuracy": metrics["accuracy"],
                "pooled_oof_precision": metrics["precision"],
                "pooled_oof_sensitivity": metrics["sensitivity"],
                "pooled_oof_specificity": metrics["specificity"],
                "pooled_oof_f1": metrics["f1"],
                "pooled_oof_tn": metrics["tn"],
                "pooled_oof_fp": metrics["fp"],
                "pooled_oof_fn": metrics["fn"],
                "pooled_oof_tp": metrics["tp"],
                "tie_pool_priority": pool_priority,
                "tie_rule_priority": rule_priority,
            })
    ranking = pd.DataFrame(ranking_rows)
    if len(ranking) != 10:
        raise RuntimeError("Expected exactly 10 candidate ensemble configurations; found {0}".format(len(ranking)))
    # The rounded AUC is used only to decide whether values are exact numerical ties.
    # A smaller pool, then the fixed rule order, resolves such ties reproducibly.
    ranking["selection_auc_rounded_12dp"] = ranking["pooled_oof_auc"].round(12)
    ranking = ranking.sort_values(
        ["selection_auc_rounded_12dp", "tie_pool_priority", "tie_rule_priority", "configuration_id"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    ranking.insert(0, "development_rank", np.arange(1, len(ranking) + 1, dtype=int))
    ranking = ranking.drop(columns=["tie_pool_priority", "tie_rule_priority"])
    all_predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["configuration_id", "patient_id"]
    ).reset_index(drop=True)
    return ranking, all_predictions, all_weights


def selection_lock(
    architecture_ranking_frame: pd.DataFrame,
    configuration_ranking_frame: pd.DataFrame,
    pools: Mapping[str, Sequence[str]],
    weights_by_configuration: Mapping[str, Mapping[str, float]],
    oof_file: Path,
    cv_manifest_file: Path,
) -> Dict[str, Any]:
    """Return a durable selection lock for the highest-ranked development configuration."""
    if configuration_ranking_frame.empty:
        raise ValueError("Configuration ranking is empty")
    selected = configuration_ranking_frame.sort_values("development_rank").iloc[0]
    configuration_id = str(selected["configuration_id"])
    selected_models = list(pools[str(selected["pool_name"])])
    return {
        "purpose": "development_only_ensemble_configuration_selection",
        "test_data_accessed": False,
        "selection_dataset": "92-patient development cohort only",
        "selection_metric": "highest pooled out-of-fold AUC",
        "tie_breakers": [
            "exact numerical AUC tie at 12 decimal places: smaller constituent pool (Top-3 before Top-5)",
            "remaining exact tie: fixed rule order Sum, Median, Max, Majority Voting, Weighted Sum",
        ],
        "candidate_pools": {name: list(values) for name, values in pools.items()},
        "candidate_rules": [{"rule_id": rule_id, "rule_name": RULE_LABELS[rule_id]} for rule_id in RULE_ORDER],
        "n_candidate_configurations": 10,
        "oof_predictions_file": str(oof_file),
        "oof_predictions_sha256": file_sha256(oof_file),
        "cv_run_manifest_file": str(cv_manifest_file),
        "cv_run_manifest_sha256": file_sha256(cv_manifest_file),
        "architecture_ranking": architecture_ranking_frame.to_dict(orient="records"),
        "selected_configuration": {
            "configuration_id": configuration_id,
            "pool_name": str(selected["pool_name"]),
            "rule_id": str(selected["rule_id"]),
            "rule_name": str(selected["rule_name"]),
            "constituent_models": selected_models,
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
        "final_test_evaluation_rule": "Apply this selected configuration only to run-matched predictions from the locked historic 61/31/46 TL benchmark; do not evaluate or choose unselected configurations on the test cohort.",
        "seed_aggregation_rule": "For patient-level inference, mean the 100 run-matched ensemble probabilities for each independent test patient before thresholding at 0.50.",
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON record atomically to secure output storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Not JSON serializable: {0}".format(type(value).__name__))
