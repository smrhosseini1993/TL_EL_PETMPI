"""Durable, compact result storage for the R1 locked final TL experiment.

The final runner writes one SQLite database transaction only after a model/seed run has
produced all required outputs. This avoids thousands of per-seed files while retaining
patient-linked, full-precision test predictions and safe resume capability.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

DATABASE_NAME = "r1_final_tl_runs.sqlite"
METRIC_NAMES = ("accuracy", "precision", "sensitivity", "specificity", "f1", "auc", "tn", "fp", "fn", "tp")
SPLIT_NAMES = ("train", "validation", "test")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def database_path(results_dir: Path) -> Path:
    return results_dir / DATABASE_NAME


def connect_results_database(results_dir: Path) -> sqlite3.Connection:
    """Open the one secure results database and create its schema if needed."""
    results_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database_path(results_dir)))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    # DELETE journaling leaves one durable database file after each committed transaction.
    # SQLite may briefly create a rollback-journal file while committing, then removes it.
    connection.execute("PRAGMA journal_mode = DELETE")
    connection.execute("PRAGMA synchronous = FULL")
    _create_schema(connection)
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS batches (
            batch_name TEXT PRIMARY KEY,
            manifest_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS run_status (
            model_name TEXT NOT NULL,
            seed INTEGER NOT NULL,
            batch_name TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed')),
            started_at TEXT NOT NULL,
            completed_at TEXT,
            elapsed_seconds REAL,
            error_message TEXT,
            PRIMARY KEY (model_name, seed)
        );

        CREATE TABLE IF NOT EXISTS run_metrics (
            model_name TEXT NOT NULL,
            seed INTEGER NOT NULL,
            batch_name TEXT NOT NULL,
            protocol_id TEXT NOT NULL,
            head_learning_rate REAL NOT NULL,
            fine_tune_learning_rate REAL NOT NULL,
            dropout_rate REAL NOT NULL,
            input_size INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            phase1_epochs INTEGER NOT NULL,
            phase2_epochs INTEGER NOT NULL,
            phase3_epochs INTEGER NOT NULL,
            early_stopping_patience INTEGER NOT NULL,
            threshold REAL NOT NULL,
            optimizer TEXT NOT NULL,
            loss TEXT NOT NULL,
            class_weights INTEGER NOT NULL,
            augmentation TEXT NOT NULL,
            elapsed_seconds REAL NOT NULL,
            train_accuracy REAL NOT NULL, train_precision REAL NOT NULL,
            train_sensitivity REAL NOT NULL, train_specificity REAL NOT NULL,
            train_f1 REAL NOT NULL, train_auc REAL NOT NULL,
            train_tn INTEGER NOT NULL, train_fp INTEGER NOT NULL,
            train_fn INTEGER NOT NULL, train_tp INTEGER NOT NULL,
            validation_accuracy REAL NOT NULL, validation_precision REAL NOT NULL,
            validation_sensitivity REAL NOT NULL, validation_specificity REAL NOT NULL,
            validation_f1 REAL NOT NULL, validation_auc REAL NOT NULL,
            validation_tn INTEGER NOT NULL, validation_fp INTEGER NOT NULL,
            validation_fn INTEGER NOT NULL, validation_tp INTEGER NOT NULL,
            test_accuracy REAL NOT NULL, test_precision REAL NOT NULL,
            test_sensitivity REAL NOT NULL, test_specificity REAL NOT NULL,
            test_f1 REAL NOT NULL, test_auc REAL NOT NULL,
            test_tn INTEGER NOT NULL, test_fp INTEGER NOT NULL,
            test_fn INTEGER NOT NULL, test_tp INTEGER NOT NULL,
            PRIMARY KEY (model_name, seed),
            FOREIGN KEY (model_name, seed) REFERENCES run_status(model_name, seed)
        );

        CREATE TABLE IF NOT EXISTS test_predictions (
            patient_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            seed INTEGER NOT NULL,
            observed_label INTEGER NOT NULL CHECK(observed_label IN (0, 1)),
            probability REAL NOT NULL CHECK(probability >= 0.0 AND probability <= 1.0),
            threshold REAL NOT NULL,
            binary_prediction INTEGER NOT NULL CHECK(binary_prediction IN (0, 1)),
            PRIMARY KEY (patient_id, model_name, seed),
            FOREIGN KEY (model_name, seed) REFERENCES run_status(model_name, seed)
        );

        CREATE TABLE IF NOT EXISTS phase_parameters (
            model_name TEXT NOT NULL,
            seed INTEGER NOT NULL,
            phase INTEGER NOT NULL CHECK(phase IN (1, 2, 3)),
            total_params INTEGER NOT NULL,
            trainable_params INTEGER NOT NULL,
            non_trainable_params INTEGER NOT NULL,
            selected_backbone_layers INTEGER NOT NULL,
            learning_rate REAL NOT NULL,
            requested_epochs INTEGER NOT NULL,
            actual_epochs INTEGER NOT NULL,
            PRIMARY KEY (model_name, seed, phase),
            FOREIGN KEY (model_name, seed) REFERENCES run_status(model_name, seed)
        );

        CREATE INDEX IF NOT EXISTS index_test_predictions_model_seed
        ON test_predictions(model_name, seed);
        CREATE INDEX IF NOT EXISTS index_test_predictions_patient
        ON test_predictions(patient_id);
        """
    )
    connection.commit()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)


def _batch_identity(manifest: Mapping[str, Any]) -> str:
    """Return the immutable scientific identity of one final-run batch.

    Local path spelling is deliberately excluded. A dry check may use a relative path
    while a later tmux launcher uses the same secure file via its absolute path; SHA-256
    fingerprints already prove the file identity. Runtime metadata is also descriptive
    and must not block a legitimate resume.
    """
    identity_fields = (
        "batch_name",
        "purpose",
        "preflight",
        "models",
        "seeds",
        "protocol",
        "split_manifest_sha256",
        "protocol_file_sha256",
        "selection_provenance_sha256",
        "split_counts",
        "split_class_counts",
        "test_data_accessed",
        "selection_performed",
        "best_seed_selection_performed",
    )
    missing = [field for field in identity_fields if field not in manifest]
    if missing:
        raise ValueError(f"Batch manifest is missing immutable identity fields: {missing}")
    return _canonical_json({field: manifest[field] for field in identity_fields})


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def initialise_results_store(
    connection: sqlite3.Connection,
    *,
    batch_name: str,
    batch_manifest: Mapping[str, Any],
    study_signature: Mapping[str, Any],
) -> None:
    """Register a compatible batch and reject mixing outputs from another study setup."""
    signature_json = _canonical_json(study_signature)
    existing = connection.execute("SELECT value FROM metadata WHERE key = 'study_signature'").fetchone()
    if existing is None:
        with connection:
            connection.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("study_signature", signature_json))
    elif existing["value"] != signature_json:
        started_run_count = connection.execute("SELECT COUNT(*) AS n FROM run_status").fetchone()["n"]
        if started_run_count:
            raise RuntimeError(
                "This results database belongs to a different code/protocol/split signature after runs began. "
                "Use a new results directory; never mix final-run outputs."
            )
        # A dry run has no model/seed record. Permit a pre-run code-only safety patch
        # (for example, an output-resume fix) to replace its signature before fitting.
        with connection:
            connection.execute("UPDATE metadata SET value = ? WHERE key = 'study_signature'", (signature_json,))

    manifest_json = _canonical_json(batch_manifest)
    current_identity = _batch_identity(batch_manifest)
    existing_batch = connection.execute("SELECT manifest_json FROM batches WHERE batch_name = ?", (batch_name,)).fetchone()
    if existing_batch is None:
        with connection:
            connection.execute(
                "INSERT INTO batches(batch_name, manifest_json, created_at) VALUES (?, ?, ?)",
                (batch_name, manifest_json, utc_now()),
            )
    else:
        existing_manifest = json.loads(existing_batch["manifest_json"])
        if _batch_identity(existing_manifest) != current_identity:
            raise RuntimeError(
                f"Batch '{batch_name}' was previously started with a different immutable manifest identity. "
                "Do not overwrite or mix its outputs."
            )
        # If a dry check used an equivalent relative path and no seed has begun, retain
        # the later full manifest so the recorded invocation mirrors the actual run.
        existing_run_count = connection.execute(
            "SELECT COUNT(*) AS n FROM run_status WHERE batch_name = ?", (batch_name,)
        ).fetchone()["n"]
        if existing_run_count == 0 and existing_batch["manifest_json"] != manifest_json:
            with connection:
                connection.execute(
                    "UPDATE batches SET manifest_json = ? WHERE batch_name = ?",
                    (manifest_json, batch_name),
                )


def run_is_complete(
    connection: sqlite3.Connection,
    model_name: str,
    seed: int,
    expected_test_patients: int,
) -> bool:
    status = connection.execute(
        "SELECT status FROM run_status WHERE model_name = ? AND seed = ?", (model_name, seed)
    ).fetchone()
    if status is None or status["status"] != "completed":
        return False
    metric_count = connection.execute(
        "SELECT COUNT(*) AS n FROM run_metrics WHERE model_name = ? AND seed = ?", (model_name, seed)
    ).fetchone()["n"]
    prediction_count = connection.execute(
        "SELECT COUNT(*) AS n FROM test_predictions WHERE model_name = ? AND seed = ?", (model_name, seed)
    ).fetchone()["n"]
    phase_count = connection.execute(
        "SELECT COUNT(*) AS n FROM phase_parameters WHERE model_name = ? AND seed = ?", (model_name, seed)
    ).fetchone()["n"]
    return metric_count == 1 and prediction_count == expected_test_patients and phase_count == 3


def mark_run_started(connection: sqlite3.Connection, *, model_name: str, seed: int, batch_name: str) -> None:
    """Start or safely restart one model/seed run after clearing incomplete residue."""
    with connection:
        connection.execute("DELETE FROM test_predictions WHERE model_name = ? AND seed = ?", (model_name, seed))
        connection.execute("DELETE FROM phase_parameters WHERE model_name = ? AND seed = ?", (model_name, seed))
        connection.execute("DELETE FROM run_metrics WHERE model_name = ? AND seed = ?", (model_name, seed))
        connection.execute(
            """
            INSERT INTO run_status(model_name, seed, batch_name, status, started_at, completed_at, elapsed_seconds, error_message)
            VALUES (?, ?, ?, 'running', ?, NULL, NULL, NULL)
            ON CONFLICT(model_name, seed) DO UPDATE SET
                batch_name = excluded.batch_name,
                status = 'running',
                started_at = excluded.started_at,
                completed_at = NULL,
                elapsed_seconds = NULL,
                error_message = NULL
            """,
            (model_name, seed, batch_name, utc_now()),
        )


def mark_run_failed(
    connection: sqlite3.Connection,
    *,
    model_name: str,
    seed: int,
    elapsed_seconds: float,
    error_message: str,
) -> None:
    with connection:
        connection.execute(
            """
            UPDATE run_status
            SET status = 'failed', completed_at = ?, elapsed_seconds = ?, error_message = ?
            WHERE model_name = ? AND seed = ?
            """,
            (utc_now(), float(elapsed_seconds), error_message[-4000:], model_name, seed),
        )


def record_completed_run(
    connection: sqlite3.Connection,
    *,
    model_name: str,
    seed: int,
    batch_name: str,
    protocol: Mapping[str, Any],
    elapsed_seconds: float,
    split_metrics: Mapping[str, Mapping[str, Any]],
    test_prediction_rows: pd.DataFrame,
    phase_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically store all final information for one completed model/seed run."""
    prediction_columns = [
        "patient_id",
        "observed_label",
        "probability",
        "threshold",
        "binary_prediction",
    ]
    if not set(prediction_columns).issubset(test_prediction_rows.columns):
        raise ValueError("Test prediction table is missing required columns")
    if test_prediction_rows["patient_id"].duplicated().any():
        raise ValueError(f"Duplicate test patient IDs for {model_name}, seed {seed}")
    probabilities = test_prediction_rows["probability"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"Invalid test probabilities for {model_name}, seed {seed}")
    if len(phase_rows) != 3 or {int(row["phase"]) for row in phase_rows} != {1, 2, 3}:
        raise ValueError(f"Expected three complete phase records for {model_name}, seed {seed}")

    metric_values: Dict[str, Any] = {
        "model_name": model_name,
        "seed": int(seed),
        "batch_name": batch_name,
        "protocol_id": protocol["protocol_id"],
        "head_learning_rate": float(protocol["head_learning_rate"]),
        "fine_tune_learning_rate": float(protocol["fine_tune_learning_rate"]),
        "dropout_rate": float(protocol["dropout_rate"]),
        "input_size": int(protocol["input_size"]),
        "batch_size": int(protocol["batch_size"]),
        "phase1_epochs": int(protocol["phase1_epochs"]),
        "phase2_epochs": int(protocol["phase2_epochs"]),
        "phase3_epochs": int(protocol["phase3_epochs"]),
        "early_stopping_patience": int(protocol["early_stopping_patience"]),
        "threshold": float(protocol["threshold"]),
        "optimizer": str(protocol["optimizer"]),
        "loss": str(protocol["loss"]),
        "class_weights": int(bool(protocol["class_weights"])),
        "augmentation": str(protocol["augmentation"]),
        "elapsed_seconds": float(elapsed_seconds),
    }
    for split in SPLIT_NAMES:
        if split not in split_metrics:
            raise ValueError(f"Missing {split} metrics for {model_name}, seed {seed}")
        for metric in METRIC_NAMES:
            value = split_metrics[split][metric]
            metric_values[f"{split}_{metric}"] = int(value) if metric in {"tn", "fp", "fn", "tp"} else float(value)

    metric_columns = list(metric_values)
    metric_sql = ", ".join(metric_columns)
    metric_placeholders = ", ".join(f":{column}" for column in metric_columns)
    prediction_rows = [
        (
            str(row.patient_id),
            model_name,
            int(seed),
            int(row.observed_label),
            float(row.probability),
            float(row.threshold),
            int(row.binary_prediction),
        )
        for row in test_prediction_rows[prediction_columns].itertuples(index=False)
    ]
    phase_values = [
        (
            model_name,
            int(seed),
            int(row["phase"]),
            int(row["total_params"]),
            int(row["trainable_params"]),
            int(row["non_trainable_params"]),
            int(row["selected_backbone_layers"]),
            float(row["learning_rate"]),
            int(row["requested_epochs"]),
            int(row["actual_epochs"]),
        )
        for row in phase_rows
    ]

    with connection:
        connection.execute(f"INSERT INTO run_metrics ({metric_sql}) VALUES ({metric_placeholders})", metric_values)
        connection.executemany(
            """
            INSERT INTO test_predictions(
                patient_id, model_name, seed, observed_label, probability, threshold, binary_prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            prediction_rows,
        )
        connection.executemany(
            """
            INSERT INTO phase_parameters(
                model_name, seed, phase, total_params, trainable_params, non_trainable_params,
                selected_backbone_layers, learning_rate, requested_epochs, actual_epochs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            phase_values,
        )
        connection.execute(
            """
            UPDATE run_status
            SET batch_name = ?, status = 'completed', completed_at = ?, elapsed_seconds = ?, error_message = NULL
            WHERE model_name = ? AND seed = ?
            """,
            (batch_name, utc_now(), float(elapsed_seconds), model_name, int(seed)),
        )


def expected_run_grid(models: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"model_name": model_name, "seed": int(seed)} for model_name in models for seed in seeds]
    )


def seed_coverage_frame(
    connection: sqlite3.Connection,
    *,
    models: Sequence[str],
    seeds: Sequence[int],
    expected_test_patients: int,
) -> pd.DataFrame:
    expected = expected_run_grid(models, seeds)
    status = pd.read_sql_query(
        "SELECT model_name, seed, batch_name, status, started_at, completed_at, elapsed_seconds, error_message FROM run_status",
        connection,
    )
    prediction_counts = pd.read_sql_query(
        "SELECT model_name, seed, COUNT(*) AS test_prediction_rows FROM test_predictions GROUP BY model_name, seed",
        connection,
    )
    phase_counts = pd.read_sql_query(
        "SELECT model_name, seed, COUNT(*) AS phase_records FROM phase_parameters GROUP BY model_name, seed",
        connection,
    )
    coverage = expected.merge(status, on=["model_name", "seed"], how="left")
    coverage = coverage.merge(prediction_counts, on=["model_name", "seed"], how="left")
    coverage = coverage.merge(phase_counts, on=["model_name", "seed"], how="left")
    coverage["test_prediction_rows"] = coverage["test_prediction_rows"].fillna(0).astype(int)
    coverage["phase_records"] = coverage["phase_records"].fillna(0).astype(int)
    coverage["status"] = coverage["status"].fillna("missing")
    coverage["coverage_status"] = np.where(
        (coverage["status"] == "completed")
        & (coverage["test_prediction_rows"] == expected_test_patients)
        & (coverage["phase_records"] == 3),
        "PASS",
        "INCOMPLETE",
    )
    return coverage.sort_values(["model_name", "seed"]).reset_index(drop=True)


def validate_results_database(
    connection: sqlite3.Connection,
    *,
    models: Sequence[str],
    seeds: Sequence[int],
    expected_test_patients: int,
) -> Dict[str, Any]:
    coverage = seed_coverage_frame(
        connection,
        models=models,
        seeds=seeds,
        expected_test_patients=expected_test_patients,
    )
    incomplete = coverage.loc[coverage["coverage_status"] != "PASS"]
    duplicate_patient_predictions = connection.execute(
        """
        SELECT COUNT(*) AS n FROM (
            SELECT patient_id, model_name, seed, COUNT(*) AS duplicate_count
            FROM test_predictions
            GROUP BY patient_id, model_name, seed
            HAVING duplicate_count > 1
        )
        """
    ).fetchone()["n"]
    invalid_probabilities = connection.execute(
        "SELECT COUNT(*) AS n FROM test_predictions WHERE probability < 0.0 OR probability > 1.0 OR probability IS NULL"
    ).fetchone()["n"]
    summary = {
        "expected_runs": int(len(coverage)),
        "complete_runs": int((coverage["coverage_status"] == "PASS").sum()),
        "incomplete_runs": int(len(incomplete)),
        "failed_runs": int((coverage["status"] == "failed").sum()),
        "running_runs": int((coverage["status"] == "running").sum()),
        "missing_runs": int((coverage["status"] == "missing").sum()),
        "duplicate_patient_prediction_groups": int(duplicate_patient_predictions),
        "invalid_probability_rows": int(invalid_probabilities),
        "valid": bool(
            len(incomplete) == 0
            and duplicate_patient_predictions == 0
            and invalid_probabilities == 0
        ),
    }
    return {"summary": summary, "coverage": coverage}


def read_metadata(connection: sqlite3.Connection) -> Dict[str, Any]:
    rows = connection.execute("SELECT key, value FROM metadata ORDER BY key").fetchall()
    return {row["key"]: json.loads(row["value"]) for row in rows}


def read_batches(connection: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT batch_name, manifest_json, created_at FROM batches ORDER BY created_at", connection)


def query_table(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    allowed = {"run_status", "run_metrics", "test_predictions", "phase_parameters"}
    if table_name not in allowed:
        raise ValueError(f"Unsupported table requested: {table_name}")
    return pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
