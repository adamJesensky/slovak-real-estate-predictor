import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent.absolute()
DATASETS_DIR = BASE_DIR / "datasets"
DOCS_DIR = BASE_DIR / "docs"
IMAGES_DIR = DOCS_DIR / "images"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for d in [DATASETS_DIR, DOCS_DIR, IMAGES_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/real_estate_db")

# Scraper Settings
SCRAPER = {
    "START_URL": "https://www.nehnutelnosti.sk",
    "DAYS_LIMIT": 7,
    "MAX_WORKERS": 10,
    "TOR_PROXY": "socks5h://127.0.0.1:9050",
    "TIMEOUT_CONNECT": 20,
    "TIMEOUT_READ": 60,
    "MAX_RETRIES": 3,
    "ROTATION_LIMIT": 50
}

# Feature Engineering
FE = {
    "MIN_PRICE": 1000,
    "MAX_PRICE_FLATS": 2000000,
    "MAX_PRICE_HOUSES": 5000000,
    "MIN_AREA": 10,
    "MAX_AREA_FLATS": 500,
    "MAX_AREA_HOUSES": 10000,
    "TARGET_ENCODING_MIN_SAMPLES": 5,
    "BRATISLAVA_COORDS": (48.1486, 17.1077)
}

# Modeling
MODELING = {
    "RANDOM_SEED": 42,
    "N_FOLDS": 10,
    "TEST_SIZE": 0.2,
    "NN_EPOCHS": 150,
    "NN_BATCH_SIZE": 64,
    "NN_LEARNING_RATE": 0.001,
    "ANOMALY_CONTAMINATION_FLATS": 0.02,
    "ANOMALY_CONTAMINATION_HOUSES": 0.03,
    "FEATURE_SELECTION_THRESHOLD": 0.005
}

# Dashboard
DASHBOARD = {
    "MAPE_BYTY": 0.0887,
    "MAPE_DOMY": 0.2514,
    "REGIONAL_CITIES": {
        'Bratislava': (48.1486, 17.1077),
        'Trnava': (48.3709, 17.5833),
        'Nitra': (48.3061, 18.0764),
        'Trenčín': (48.8945, 18.0444),
        'Žilina': (49.2231, 18.7394),
        'Banská Bystrica': (48.7363, 19.1462),
        'Prešov': (49.0018, 21.2393),
        'Košice': (48.7164, 21.2611),
    },
    "NN_CLAMP_MIN": 6.5,
    "NN_CLAMP_MAX": 16.0,
    "NN_FALLBACK": 11.5,
}

# Web App
WEB_APP = {
    "TITLE": "Real Estate AI Valuator",
    "ICON": "🏠",
    "LAYOUT": "wide"
}
