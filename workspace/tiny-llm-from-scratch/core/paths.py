from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"

RAW_DATA_DIR = DATASETS_DIR / "raw"

PROCESSED_DATA_DIR = DATASETS_DIR / "processed"

TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"

MODEL_DIR = PROJECT_ROOT / "model"

TRAINING_DIR = PROJECT_ROOT / "training"

INFERENCE_DIR = PROJECT_ROOT / "inference"

CONFIG_DIR = PROJECT_ROOT / "configs"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

LOG_DIR = PROJECT_ROOT / "logs"

REPORT_DIR = PROJECT_ROOT / "reports"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"