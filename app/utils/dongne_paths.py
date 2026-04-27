from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DONGNE_DATA_DIR = PROJECT_ROOT / "data" / "dongne"
DONGNE_RAW_DATA_DIR = DONGNE_DATA_DIR / "raw"
DONGNE_PROCESSED_DATA_DIR = DONGNE_DATA_DIR / "processed"
DONGNE_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "dongne"
