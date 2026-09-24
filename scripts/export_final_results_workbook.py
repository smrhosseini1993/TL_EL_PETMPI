#!/usr/bin/env python3
"""Validate one compact R1 TL SQLite result store and export a readable Excel workbook.

The database remains the source of truth. The workbook is created only from validated
records and is intended for secure human inspection and downstream manuscript work.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import pandas as pd
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from final_results_store import (
    DATABASE_NAME,
    connect_results_database,
    query_table,
    read_batches,
    read_metadata,
    seed_coverage_frame,
    validate_results_database,
)
from tl_reanalysis_core import SUPPORTED_MODELS


DEFAULT_SEEDS = list(range(1, 101))


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seed_specification(value: str) -> List[int]:
    values: List[int] = []
    for token in parse_csv(value):
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid seed range: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and export compact R1 final TL results.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Secure R1_final_TL directory containing the SQLite database.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Workbook destination. Defaults to R1_TL_final_results.xlsx inside --results-dir.",
    )
    parser.add_argument("--models", default=",".join(SUPPORTED_MODELS), help="Expected models; default is all 11.")
    parser.add_argument("--seeds", default="1-100", help="Expected seeds; default is 1-100.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Export a clearly labelled interim workbook; normally exports stop unless all expected runs are valid.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing workbook.")
    return parser.parse_args()


def style_sheet(worksheet) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for index, column_cells in enumerate(worksheet.columns, start=1):
        sample_lengths = [len(str(cell.value)) for cell in list(column_cells)[:250] if cell.value is not None]
        width = min(max(sample_lengths, default=10) + 2, 45)
        worksheet.column_dimensions[get_column_letter(index)].width = width


def flatten_mapping(prefix: str, value: Any, rows: List[Mapping[str, str]]) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_mapping(nested_prefix, nested_value, rows)
    else:
        rows.append({"item": prefix, "value": json.dumps(value) if isinstance(value, (list, tuple)) else str(value)})


def coverage_by_model(coverage: pd.DataFrame, expected_test_patients: int) -> pd.DataFrame:
    rows = []
    for model_name, group in coverage.groupby("model_name", sort=True):
        missing = group.loc[group["coverage_status"] != "PASS", "seed"].tolist()
        rows.append(
            {
                "model_name": model_name,
                "expected_seeds": int(len(group)),
                "completed_seeds": int((group["coverage_status"] == "PASS").sum()),
                "missing_or_invalid_seeds": ", ".join(map(str, missing)) if missing else "",
                "test_prediction_rows": int(group["test_prediction_rows"].sum()),
                "expected_test_prediction_rows": int(len(group) * expected_test_patients),
                "status": "PASS" if not missing else "INCOMPLETE",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    models = parse_csv(args.models)
    if set(models) - set(SUPPORTED_MODELS):
        raise ValueError("--models contains an unsupported architecture")
    seeds = parse_seed_specification(args.seeds)
    expected_test_patients = 46
    output = args.output or (args.results_dir / "R1_TL_final_results.xlsx")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Workbook already exists: {output}. Use --overwrite only after review.")

    connection = connect_results_database(args.results_dir)
    try:
        validation = validate_results_database(
            connection,
            models=models,
            seeds=seeds,
            expected_test_patients=expected_test_patients,
        )
        summary = validation["summary"]
        coverage = validation["coverage"]
        print(json.dumps(summary, indent=2, sort_keys=True))
        if not summary["valid"] and not args.allow_incomplete:
            raise RuntimeError("Results are incomplete or invalid; no final workbook was exported.")

        run_metrics = query_table(connection, "run_metrics").sort_values(["model_name", "seed"])
        test_predictions = query_table(connection, "test_predictions").sort_values(["model_name", "seed", "patient_id"])
        phase_parameters = query_table(connection, "phase_parameters").sort_values(["model_name", "seed", "phase"])
        run_status = query_table(connection, "run_status").sort_values(["model_name", "seed"])
        metadata = read_metadata(connection)
        batches = read_batches(connection)
    finally:
        connection.close()

    metadata_rows: List[Mapping[str, str]] = [
        {"item": "database", "value": str(args.results_dir / DATABASE_NAME)},
        {"item": "workbook_status", "value": "FINAL_VALIDATED" if summary["valid"] else "INTERIM_INCOMPLETE"},
        {"item": "expected_models", "value": ", ".join(models)},
        {"item": "expected_seeds", "value": ", ".join(map(str, seeds))},
        {"item": "expected_test_patients_per_run", "value": str(expected_test_patients)},
    ]
    for key, value in metadata.items():
        flatten_mapping(key, value, metadata_rows)
    for key, value in summary.items():
        metadata_rows.append({"item": f"validation.{key}", "value": str(value)})
    metadata_frame = pd.DataFrame(metadata_rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        run_metrics.to_excel(writer, sheet_name="Run_metrics_1100", index=False)
        test_predictions.to_excel(writer, sheet_name="Test_predictions", index=False)
        coverage_by_model(coverage, expected_test_patients).to_excel(writer, sheet_name="Seed_coverage", index=False)
        metadata_frame.to_excel(writer, sheet_name="Protocol_and_split", index=False)
        phase_parameters.to_excel(writer, sheet_name="Phase_parameters", index=False)
        run_status.to_excel(writer, sheet_name="Run_status", index=False)
        batches.to_excel(writer, sheet_name="Batches", index=False)
        for worksheet in writer.book.worksheets:
            style_sheet(worksheet)
    print(f"Exported workbook: {output}")


if __name__ == "__main__":
    main()
