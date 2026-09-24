#!/usr/bin/env python3
"""Synthetic no-fit tests for R1 runner manifests and CLI safeguards.

No real patient images or labels are used. Placeholder JPEG files are never decoded
because both invoked runners use --dry-run.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="r1_tl_dryrun_") as temp:
        root = Path(temp) / "secure_project"
        training = root / "data" / "training"
        testing = root / "data" / "test"
        training.mkdir(parents=True)
        testing.mkdir(parents=True)
        labels = [0] * 56 + [1] * 36
        (training / "ica_lables.txt").write_text("\n".join(map(str, labels)) + "\n", encoding="utf-8")
        for index in range(92):
            (training / f"development_{index:03d}.jpg").touch()

        cv_output = Path(temp) / "cv_output"
        run(
            [
                sys.executable,
                "TL_crossvalidation.py",
                "--root",
                str(root),
                "--output-dir",
                str(cv_output),
                "--models",
                "Xception",
                "--protocol-ids",
                "hlr1e-04_flr3e-06_do50",
                "--allow-partial",
                "--dry-run",
            ]
        )
        fold_manifest = pd.read_csv(cv_output / "manifests" / "cv_fold_manifest.csv")
        assert len(fold_manifest) == 92 and not fold_manifest["patient_id"].duplicated().any()
        assert not (cv_output / "selected_protocol.json").exists()

        rows = []
        for index in range(61):
            name = f"train_{index:03d}.jpg"
            (training / name).touch()
            rows.append({"patient_id": f"train_{index:03d}", "relative_path": f"data/training/{name}", "split": "train", "observed_label": 0 if index < 36 else 1})
        for index in range(31):
            name = f"validation_{index:03d}.jpg"
            (training / name).touch()
            rows.append({"patient_id": f"validation_{index:03d}", "relative_path": f"data/training/{name}", "split": "validation", "observed_label": 0 if index < 20 else 1})
        for index in range(46):
            name = f"test_{index:03d}.jpg"
            (testing / name).touch()
            rows.append({"patient_id": f"test_{index:03d}", "relative_path": f"data/test/{name}", "split": "test", "observed_label": 0 if index < 26 else 1})
        split_manifest = Path(temp) / "fixed_split.csv"
        pd.DataFrame(rows).to_csv(split_manifest, index=False)

        protocol_dir = Path(temp) / "complete_cv"
        protocol_dir.mkdir()
        protocol = {"protocol_id": "hlr1e-04_flr3e-06_do50", "head_learning_rate": 1e-4, "fine_tune_learning_rate": 3e-6, "dropout_rate": 0.5}
        (protocol_dir / "selected_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        (protocol_dir / "selection_provenance.json").write_text(
            json.dumps({"test_data_accessed": False, "selected_protocol": protocol}), encoding="utf-8"
        )
        final_results = Path(temp) / "R1_final_TL"
        run(
            [
                sys.executable,
                "TL_fixedvalidation.py",
                "--root",
                str(root),
                "--split-manifest",
                str(split_manifest),
                "--protocol-file",
                str(protocol_dir / "selected_protocol.json"),
                "--results-dir",
                str(final_results),
                "--batch-name",
                "seeds_001_005",
                "--models",
                "Xception",
                "--seeds",
                "1-5",
                "--preflight",
                "--dry-run",
            ]
        )
        assert (final_results / "r1_final_tl_runs.sqlite").is_file()
        assert (final_results / "run_manifest_seeds_001_005.json").is_file()
        print("Synthetic runner dry-run checks passed.")


if __name__ == "__main__":
    main()
