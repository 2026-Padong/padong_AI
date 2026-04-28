from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNC_LOG_SCRIPT = PROJECT_ROOT / "scripts" / "recommendation" / "sync_backend_logs.py"
BUILD_DATASET_SCRIPT = PROJECT_ROOT / "scripts" / "recommendation" / "build_pair_dataset.py"
TRAIN_MODEL_SCRIPT = PROJECT_ROOT / "scripts" / "recommendation" / "train_lgbm_regressor.py"


def run_step(script_path: Path) -> None:
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=PROJECT_ROOT)


def main() -> None:
    run_step(SYNC_LOG_SCRIPT)
    run_step(BUILD_DATASET_SCRIPT)
    run_step(TRAIN_MODEL_SCRIPT)


if __name__ == "__main__":
    main()
