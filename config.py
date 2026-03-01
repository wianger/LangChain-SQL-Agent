import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── API ──────────────────────────────────────────────────────────────────────
API_KEY = "API_KEY_HERE"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_NAME = "doubao-seed-2-0-pro-260215"
EMBEDDING_MODEL = "doubao-embedding-vision-251215"

# ── Dataset paths ────────────────────────────────────────────────────────────
LOCOMO_DATA_PATH = "./datas/locomo/data/locomo10.json"
SYLLABUSQA_TEST_PATH = "./datas/SyllabusQA/data/dataset_split/"
SYLLABI_TEXT_DIR = "./datas/SyllabusQA/syllabi/syllabi_redacted/text"
SYLLABI_META_PATH = "./datas/SyllabusQA/syllabi/syllabi_meta_info.csv"

FINANCEBENCH_QA_PATH = "./datas/financebench/data/financebench_open_source.jsonl"
FINANCEBENCH_DOC_INFO_PATH = (
    "./datas/financebench/data/financebench_document_information.jsonl"
)
FINANCEBENCH_PDF_DIR = "./datas/financebench/pdfs"

QASPER_TEST_PATH = "./datas/qasper/test.json"

CLAPNQ_DATA_DIR = "./datas/clapnq/annotated_data"

# ── Database paths ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOCOMO_DB = os.path.join(DATA_DIR, "locomo.db")
SYLLABUSQA_DB = os.path.join(DATA_DIR, "syllabusqa.db")
FINANCEBENCH_DB = os.path.join(DATA_DIR, "financebench.db")
QASPER_DB = os.path.join(DATA_DIR, "qasper.db")
CLAPNQ_DB = os.path.join(DATA_DIR, "clapnq.db")

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
