from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "Musical_Instruments_5.json.gz"
PROCESSED = ROOT / "data" / "processed"
REVIEWS_PARQUET = PROCESSED / "reviews.parquet"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLFLOW_URI = "sqlite:///" + (ROOT / "mlflow.db").as_posix()

SEED = 42
LABELS = ["negative", "neutral", "positive"]   # 1-2 stars, 3 stars, 4-5 stars
TEST_FROM = "2017-01-01"                       # time-based split: train on the past, test on the future

# Shared evaluation sample so slow transformer models and fast baselines are
# scored on exactly the same reviews.
EVAL_SAMPLE = 6000
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINETUNE_BASE = "distilbert-base-uncased"
FINETUNE_TRAIN_PER_CLASS = 2000   # CPU/RAM budget on a 16 GB laptop
MAX_LEN = 128
