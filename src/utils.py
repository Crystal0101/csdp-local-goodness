"""Process isolation and short display messages; raw logs stay on disk."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import subprocess
import sys

from .config import ROOT


def fresh_directory(label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return ROOT / "fresh_runs" / f"{label}_{stamp}"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def run_logged(arguments: list, log_file: Path, python: str | None = None) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    interpreter = python or sys.executable
    environment = {
        **os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    with log_file.open("w") as log:
        completed = subprocess.run(
            [interpreter, *map(str, arguments)], stdout=log,
            stderr=subprocess.STDOUT, env=environment, check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Process failed with status {completed.returncode}; "
            f"inspect {display_path(log_file)}"
        )


def probability_digest(probabilities: list) -> str:
    import hashlib
    encoded = json.dumps(probabilities, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def clean_display_paths(value):
    if isinstance(value, str):
        return re.sub(r'/(?:Users|home)/[^\s"<>]+', "<local-path>", value)
    if isinstance(value, list):
        return [clean_display_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_display_paths(item) for key, item in value.items()}
    return value
