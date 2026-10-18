"""Paths, seeds, and label settings shared by the notebooks.

Paths are relative to the project root unless an environment variable
provides an absolute path or a project-relative override. The raw extract
is never renamed or rewritten by this module.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# Folder that contains src/, notebooks/, data/, and requirements.txt.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env without overriding existing variables.

    A small reader is used so the project does not need python-dotenv.
    Blank lines and comments are ignored. Quoted values are unquoted.
    """
    env_path = PROJECT_ROOT / ".env" if path is None else path
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _env_path(name: str, default: Path) -> Path:
    """Resolve a directory from the environment, or return the project default."""
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


load_env_file()

# Data directories. Raw files stay as downloaded. Interim and processed
# files are created by later notebooks and are not a substitute for raw.
DATA_DIR = _env_path("DATA_DIR", PROJECT_ROOT / "data")
RAW_DATA_DIR = _env_path("DATA_RAW_DIR", DATA_DIR / "raw")
INTERIM_DATA_DIR = _env_path("DATA_INTERIM_DIR", DATA_DIR / "interim")
PROCESSED_DATA_DIR = _env_path("DATA_PROCESSED_DIR", DATA_DIR / "processed")

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = _env_path("MODELS_DIR", PROJECT_ROOT / "models")
APP_DIR = PROJECT_ROOT / "app"
RESULTS_DIR = _env_path("RESULTS_DIR", PROJECT_ROOT / "results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = _env_path("REPORTS_DIR", PROJECT_ROOT / "reports")
TESTS_DIR = PROJECT_ROOT / "tests"

# Optional single-file override when data/raw holds more than one download.
RAW_DATA_FILE = os.environ.get("DATA_RAW_FILE", "").strip() or None

# One seed for later splits, samplers, and estimators.
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

# Classes predicted at the time a request is logged.
TARGET_LABELS = ("Resolved Fast", "Resolved Slow", "Unresolved")
TARGET_COLUMN = "resolution_outcome"

# Later evaluation pays particular attention to these two classes.
RECALL_FOCUS_LABELS = ("Resolved Slow", "Unresolved")

# Names as written in the project brief. The feature service uses the
# underscored field names; a CSV download may use either spelling.
PLANNED_CREATED_DATE_COLUMN = "Created On Date"
PLANNED_COMPLETED_DATE_COLUMN = "Completed Date"
PLANNED_CHANGED_ON_COLUMN = "Changed on"
SERVICE_CREATED_DATE_COLUMN = "Created_On_Date"
SERVICE_COMPLETED_DATE_COLUMN = "Completed_Date"
SERVICE_CHANGED_ON_COLUMN = "Changed_on"

# Known after logging, or constructed from those fields. Never use them as inputs.
LEAKAGE_OR_POST_EVENT_COLUMNS = (
    PLANNED_COMPLETED_DATE_COLUMN,
    PLANNED_CHANGED_ON_COLUMN,
    SERVICE_COMPLETED_DATE_COLUMN,
    SERVICE_CHANGED_ON_COLUMN,
)
CONSTRUCTED_COLUMNS = (
    TARGET_COLUMN,
    "resolution_days",
    "label_status",
    "duration_flag",
    "coordinate_quality",
    "notification_id_repeated",
    "notification_type_matches_id",
)

# Stated in the project plan. The published layer's own dates are in
# reports/data_source.md and are not the same end date.
DATASET_OWNER = "City of Cape Town"
DATASET_NAME = "SAP C3 Notifications"
DATASET_PORTAL = "City of Cape Town Open Data Portal"
DATASET_TITLE = "Service Requests 2023 until  06 October 2026 (Deprecated)"
PLANNED_COLLECTION_PERIOD = "2023 until 18 September 2026"
PLANNED_COLLECTION_END = date(2026, 9, 18)
SOURCE_PAGE_URL = (
    "https://odp-cctegis.opendata.arcgis.com/datasets/"
    "90711027d33940b5a06e96ad8a8f7ede_0/explore"
)
# Link stored on the ArcGIS item. It redirects to the City terms page.
LICENCE_PAGE_URL = "https://www.capetown.gov.za/General/Terms-of-use-open-data"
FEATURE_LAYER_URL = (
    "https://services6.arcgis.com/nyYfO9SxHU2ChQd9/arcgis/rest/services/"
    "Service_Requests_2023_until_20_May_2026/FeatureServer/0"
)

# Seven calendar days is a provisional cut, not a fitted result and not a
# citywide resolution standard. The City has said C3 turnaround depends on
# the department. Override either value in the environment if the team
# adopts a different documented rule. The observation window defaults to
# the same number so an open request is Unresolved only after it has had
# the full fast-resolution period to be completed.
PROVISIONAL_THRESHOLD_DAYS = 7
UNUSUAL_DURATION_DAYS = 365


def _positive_int(name: str, default: int) -> int:
    """Read a positive integer, or return the documented default."""
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive whole number. Got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0. Got {value}.")
    return value


FAST_SLOW_THRESHOLD_DAYS = _positive_int("FAST_SLOW_THRESHOLD_DAYS", PROVISIONAL_THRESHOLD_DAYS)
OBSERVATION_WINDOW_DAYS = _positive_int("OBSERVATION_WINDOW_DAYS", FAST_SLOW_THRESHOLD_DAYS)

# Identifiers and precise location. They are kept in the raw extract and
# blocked from the default model-feature list.
IDENTIFIER_COLUMNS = (
    "Notification",
    "Notification_type",
    "ObjectId",
)
PRECISE_LOCATION_COLUMNS = (
    "X_Y_Co_ordinate_1",
    "X_Y_Co_ordinate_2",
)
# Envelope counted on the published layer on 16 October 2026.
# Coordinate 1 is the easting-like value and coordinate 2 the northing-like
# value. This is not an official EPSG declaration; only a handful of rows
# fall in a longitude/latitude box. Rows outside the envelope are flagged,
# not deleted.
LOCAL_GRID_X_MIN = -200_000
LOCAL_GRID_X_MAX = 200_000
LOCAL_GRID_Y_MIN = 3_600_000
LOCAL_GRID_Y_MAX = 4_000_000


def required_directories() -> tuple[Path, ...]:
    """Directories the repository is expected to contain."""
    return (
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        NOTEBOOKS_DIR,
        MODELS_DIR,
        APP_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        REPORTS_DIR,
        TESTS_DIR,
        PROJECT_ROOT / "src",
    )


def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """Set Python, NumPy, and PyTorch seeds. This does not train a model."""
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
