import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── API ──────────────────────────────────────────────────────────────────────
API_KEY = "API_KEY_HERE"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_NAME = "doubao-seed-1-8-251228"
EMBEDDING_MODEL = "doubao-embedding-vision-250615"

# ── Dataset paths ────────────────────────────────────────────────────────────
LOCOMO_DATA_PATH = "./datas/locomo/data/locomo10.json"
SYLLABUSQA_TEST_PATH = "./datas/SyllabusQA/data/dataset_split/test.json"
SYLLABI_TEXT_DIR = "./datas/SyllabusQA/syllabi/syllabi_redacted/text"
SYLLABI_META_PATH = "./datas/SyllabusQA/syllabi/syllabi_meta_info.csv"

# ── Database paths ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOCOMO_DB = os.path.join(DATA_DIR, "locomo.db")
SYLLABUSQA_DB = os.path.join(DATA_DIR, "syllabusqa.db")

# ── Results ──────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Tiktoken ─────────────────────────────────────────────────────────────────
TIKTOKEN_ENCODING = "cl100k_base"

# ── Experiment ───────────────────────────────────────────────────────────────
SMALL_SAMPLE_SIZE = 20
AGENT_MAX_ITERATIONS = 15
CONCURRENCY = 10

# LoCoMo category mapping
LOCOMO_CATEGORIES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}
