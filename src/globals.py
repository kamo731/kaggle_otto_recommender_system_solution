from pathlib import Path

# dir
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
LOGS = ROOT / "logs"
SUBMISSION = ROOT / "output" / "sub"
OOF = ROOT / "output" / "oof"
ENSEMBLE = ROOT / "output" / "sub" / "ensemble"
CO_VISITATION_MATRIX = ROOT / "output" / "01_co_visitation_matrix"
CANDIDATES = ROOT / "output" / "02_candidates"
FEATURES = ROOT / "output" / "03_features"
DATASET = ROOT / "output" / "dataset"


# file
TRAIN_ORIGINAL = INPUT / "otto-full-optimized-memory-footprint/train.parquet"
TEST_ORIGINAL = INPUT / "otto-full-optimized-memory-footprint/test.parquet"
TRAIN_STEP1 = INPUT / "otto-train-and-test-data-for-local-validation" / "train.parquet"
TRAIN_STEP2 = INPUT / "otto-train-and-test-data-for-local-validation" / "test.parquet"
TRAIN_STEP2_LABEL = INPUT / "otto-train-and-test-data-for-local-validation" / "test_labels.parquet"


TYPE2ID = {"clicks": 0, "carts": 1, "orders": 2}
ID2TYPE = {0: "clicks", 1: "carts", 2: "orders"}
