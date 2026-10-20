"""Execute notebooks 01–07 without writing over the saved project outputs.

The copies land in evidence/reproducibility/notebooks/. Results, models,
reports, and the processed parquet written by those copies use that
directory. The raw extract is read from data/raw/. The locked-test files
already stored under results/tables/ are not opened for replacement.
"""

from __future__ import annotations

import os
import shutil
import sys
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "notebooks"
DESTINATION = ROOT / "evidence" / "reproducibility" / "notebooks"
ISOLATED = DESTINATION / "isolated"
NOTEBOOKS = (
    "01_data_acquisition_and_audit.ipynb",
    "02_exploratory_data_analysis.ipynb",
    "03_preprocessing_and_baselines.ipynb",
    "04_traditional_machine_learning.ipynb",
    "05_neural_networks.ipynb",
    "06_evaluation_and_optimisation.ipynb",
    "07_ablation_and_statistical_testing.ipynb",
)


def _prepare_environment() -> None:
    os.environ["RESULTS_DIR"] = str(ISOLATED / "results")
    os.environ["MODELS_DIR"] = str(ISOLATED / "models")
    os.environ["DATA_PROCESSED_DIR"] = str(ISOLATED / "processed")
    os.environ["DATA_INTERIM_DIR"] = str(ISOLATED / "interim")
    os.environ["REPORTS_DIR"] = str(ISOLATED / "reports")
    for name in ("RESULTS_DIR", "MODELS_DIR", "DATA_PROCESSED_DIR", "DATA_INTERIM_DIR", "REPORTS_DIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


def execute_one(name: str) -> Path:
    source = SOURCE_DIR / name
    executed = DESTINATION / "executed" / name
    executed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, executed)
    notebook = nbformat.read(executed, as_version=4)
    client = NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=60 * 60 * 6,
        resources={"metadata": {"path": str(ROOT)}},
    )
    os.chdir(ROOT)
    client.execute()
    nbformat.write(notebook, executed)
    return executed


def main() -> None:
    _prepare_environment()
    start = sys.argv[1] if len(sys.argv) > 1 else NOTEBOOKS[0]
    if start not in NOTEBOOKS:
        raise SystemExit(f"Unknown notebook: {start}")
    selected = NOTEBOOKS[NOTEBOOKS.index(start) :]
    log_path = DESTINATION / "execution_log.txt"
    completed = []
    mode = "a" if start != NOTEBOOKS[0] else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        log.write(f"Isolated notebook execution from {start}\n")
        for key, value in sorted(os.environ.items()):
            if key.endswith("_DIR") or key in {"DATA_RAW_FILE", "RANDOM_SEED"}:
                log.write(f"{key}={value}\n")
        log.write("\n")
        for name in selected:
            log.write(f"START {name}\n")
            log.flush()
            try:
                path = execute_one(name)
            except Exception:
                log.write(f"FAILED {name}\n")
                log.write(traceback.format_exc())
                log.write(f"COMPLETED {', '.join(completed) if completed else '(none)'}\n")
                log.flush()
                print(f"FAILED {name}", file=sys.stderr)
                raise
            completed.append(name)
            log.write(f"FINISHED {path}\n")
            log.flush()
            print(f"FINISHED {name}")
        log.write("ALL_REQUESTED_NOTEBOOKS_FINISHED\n")


if __name__ == "__main__":
    main()
