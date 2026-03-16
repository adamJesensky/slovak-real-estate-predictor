
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import torch
import torch.nn as nn
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import shap
import matplotlib.pyplot as plt
import pydeck as pdk
import sys
import unicodedata

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ASSETS_DIR = os.path.join(BASE_DIR, 'dashboard', 'assets')

sys.path.insert(0, os.path.join(BASE_DIR))
from config import DASHBOARD

# --- Neural Network Definition (Must match training) ---
class AdvancedNN(nn.Module):
    def __init__(self, input_dim):
        super(AdvancedNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

# --- Load Resources ---
@st.cache_resource
def load_resources(category):
    with open(os.path.join(ASSETS_DIR, f'mappings_{category}.json'), 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    features = joblib.load(os.path.join(MODELS_DIR, f'features_{category}.joblib'))
    models = {}
    models['xgb'] = joblib.load(os.path.join(MODELS_DIR, f'xgb_{category}.joblib'))
    models['lgb'] = joblib.load(os.path.join(MODELS_DIR, f'lgb_{category}.joblib'))
    cb = CatBoostRegressor()
    cb.load_model(os.path.join(MODELS_DIR, f'cb_{category}.cbm'))
    models['cb'] = cb
    models['meta'] = joblib.load(os.path.join(MODELS_DIR, f'meta_{category}.joblib'))
    imputer = joblib.load(os.path.join(MODELS_DIR, f'nn_imputer_{category}.joblib'))
    scaler = joblib.load(os.path.join(MODELS_DIR, f'nn_scaler_{category}.joblib'))
    device = torch.device('cpu')
    nn_model = AdvancedNN(len(features))
    nn_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'nn_{category}.pth'), map_location=device))
    nn_model.eval()
    models['nn'] = nn_model
    models['nn_imputer'] = imputer
    models['nn_scaler'] = scaler
    return mappings, features, models

# --- Helpers ---

def calculate_confidence_interval(prediction, category):
    mape = DASHBOARD['MAPE_BYTY'] if category == 'byty' else DASHBOARD['MAPE_DOMY']
    return prediction * (1 - mape), prediction * (1 + mape)

def predict_ensemble(X_input, models):
    """Run all 4 models + meta-learner. Returns (final_price, individual_log_preds)."""
    pred_xgb = models['xgb'].predict(X_input)
    pred_lgb = models['lgb'].predict(X_input)
    pred_cb = models['cb'].predict(X_input)
    X_nn = models['nn_scaler'].transform(models['nn_imputer'].transform(X_input))
    with torch.no_grad():
        pred_nn = models['nn'](torch.FloatTensor(X_nn)).numpy().flatten()
        pred_nn = np.nan_to_num(pred_nn, nan=DASHBOARD['NN_FALLBACK'],
                                posinf=DASHBOARD['NN_CLAMP_MAX'], neginf=DASHBOARD['NN_CLAMP_MIN'])
        pred_nn = np.clip(pred_nn, DASHBOARD['NN_CLAMP_MIN'], DASHBOARD['NN_CLAMP_MAX'])
    stack_X = np.column_stack((pred_xgb, pred_lgb, pred_cb, pred_nn))
    final_log = models['meta'].predict(stack_X)
    final_price = np.exp(final_log)[0]
    return final_price, (pred_xgb, pred_lgb, pred_cb, pred_nn)

FEATURE_LABELS = {
    'location_score_m2': 'Lokalita (cena/m\u00b2)',
    'floor_size': 'Plocha',
    'log_area': 'Plocha',
    'condition_score': 'Stav nehnute\u013enosti',
    'dist_bratislava': 'Vzdialenos\u0165 od Bratislavy',
    'dist_nearest_city': 'Vzdialenos\u0165 od krajsk\u00e9ho mesta',
    'room_count': 'Po\u010det izieb',
    'has_lift': 'V\u00fd\u0165ah',
    'relative_floor': 'Relat\u00edvne poschodie',
    'utilities_score': 'In\u017e. siete (N/A)',
    'has_balcony': 'Balk\u00f3n',
    'has_loggia': 'Loggia',
    'has_cellar': 'Pivnica',
    'has_garage': 'Gar\u00e1\u017e',
    'has_parking': 'Parkovanie',
    'has_terrace': 'Terasa',
    'land_area': 'Plocha pozemku',
    'log_land_area': 'Plocha pozemku',
    'built_up_area': 'Zastavan\u00e1 plocha',
    'built_up_ratio': 'Pomer zastavanej plochy',
    'avg_room_size': 'Priemern\u00e1 ve\u013ekos\u0165 izby',
    'current_floor': 'Poschodie',
    'total_floors': 'Po\u010det poschod\u00ed',
    'no_lift_high_floor': 'Vysok\u00e9 poschodie bez v\u00fd\u0165ahu',
    'is_ground_floor': 'Pr\u00edzemie',
    'is_top_floor': 'Posledn\u00e9 poschodie',
    'balkon': 'Balk\u00f3n (po\u010det)',
    'loggia': 'Loggia (po\u010det)',
    'podlazie': 'Podla\u017eie',
    'days_on_market': 'Dni na trhu',
    'has_gas': 'Plyn',
    'has_water': 'Voda',
    'has_electricity': 'Elektrina',
    'has_sewerage': 'Kanaliz\u00e1cia',
    'has_pantry': '\u0160pajza',
    'has_warehouse': 'Sklad',
    'has_ac': 'Klimatiz\u00e1cia',
    'month_sin': 'Sez\u00f3nnos\u0165',
    'month_cos': 'Sez\u00f3nnos\u0165',
    'year_added': 'Rok inzer\u00e1tu',
    'month_added': 'Mesiac inzer\u00e1tu',
}

def shap_to_eur(shap_value, final_price):
    """Convert SHAP value (ln-price space) to EUR impact."""
    return final_price * (1 - np.exp(-shap_value))

def _format_feature_value(fname, raw_value):
    """Format a feature's raw value for display next to SHAP explanation."""
    if fname in ('location_score_m2',):
        return f"{raw_value:,.0f} \u20ac/m\u00b2"
    if fname in ('floor_size', 'land_area', 'built_up_area'):
        return f"{raw_value:,.0f} m\u00b2"
    if fname in ('log_area', 'log_land_area'):
        return f"{np.expm1(raw_value):,.0f} m\u00b2"
    if fname in ('dist_bratislava', 'dist_nearest_city'):
        return f"{raw_value:,.0f} km"
    if fname in ('room_count', 'current_floor', 'total_floors', 'balkon', 'loggia', 'podlazie'):
        return f"{int(raw_value)}"
    if fname == 'condition_score':
        score_labels = {0: 'In\u00fd', 1: 'P\u00f4vodn\u00fd', 2: '\u010ciasto\u010dn\u00e1 rek.', 3: 'Kompletn\u00e1 rek.', 4: 'Novostavba'}
        return score_labels.get(int(raw_value), str(int(raw_value)))
    if fname == 'relative_floor':
        return f"{raw_value:.0%}"
    if fname == 'avg_room_size':
        return f"{raw_value:,.0f} m\u00b2"
    if fname in ('days_on_market',):
        return f"{int(raw_value)} dn\u00ed"
    if fname == 'built_up_ratio':
        return f"{raw_value:.0%}"
    # Binary features
    if fname.startswith('has_') or fname in ('is_ground_floor', 'is_top_floor', 'no_lift_high_floor'):
        return "\u00e1no" if raw_value >= 0.5 else "nie"
    # One-hot encoded categories — extract the category name
    for prefix in ('stav_final_', 'construction_type_mapped_', 'heating_type_', 'vlastnictvo_'):
        if fname.startswith(prefix):
            return "\u00e1no" if raw_value >= 0.5 else "nie"
    return None

def get_plain_language_shap(shap_values_row, final_price, X_input=None, top_n=5):
    """Return list of (feature_label, eur_impact, formatted_value) sorted by |impact|."""
    values = shap_values_row.values
    feature_names = shap_values_row.feature_names
    impacts = []
    for sv, fname in zip(values, feature_names):
        eur = shap_to_eur(sv, final_price)
        label = FEATURE_LABELS.get(fname, fname)
        # Get actual value for display
        fmt_val = None
        if X_input is not None:
            try:
                raw_val = X_input[fname].values[0] if hasattr(X_input, '__getitem__') else None
                if raw_val is not None:
                    fmt_val = _format_feature_value(fname, raw_val)
            except (KeyError, IndexError):
                pass
        impacts.append((label, eur, fname, fmt_val))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    seen_labels = set()
    unique = []
    for label, eur, fname, fmt_val in impacts:
        if label not in seen_labels:
            seen_labels.add(label)
            unique.append((label, eur, fmt_val))
        if len(unique) >= top_n:
            break
    return unique

def get_shap_diff_top(shap_original, shap_comparison, price_comparison, top_n=3):
    """Return top N SHAP diff factors between two predictions."""
    diff_values = shap_comparison.values - shap_original.values
    feature_names = shap_comparison.feature_names
    diffs = []
    seen = set()
    sorted_pairs = sorted(zip(diff_values, feature_names), key=lambda x: abs(x[0]), reverse=True)
    for sv, fn in sorted_pairs:
        lbl = FEATURE_LABELS.get(fn, fn)
        if lbl not in seen:
            seen.add(lbl)
            eur_diff = price_comparison * (1 - np.exp(-sv))
            diffs.append((lbl, eur_diff))
        if len(diffs) >= top_n:
            break
    return diffs

MAP_STYLES = {
    "Dark": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    "Light": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
}

def strip_diacritics(text):
    """Remove diacritics: '\u017dilina' -> 'Zilina', 'Pre\u0161ov' -> 'Presov'."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.category(c).startswith('M'))

@st.cache_resource
def get_shap_explainer(_model, model_name):
    """Cache SHAP TreeExplainer to avoid recreating on every prediction."""
    return shap.TreeExplainer(_model)

# --- Feature Engineering Helper ---
def process_input(data, mappings, feature_cols, category):
    df = pd.DataFrame([data])

    loc_stats = mappings['locations'].get(data['obec_cast'], {'location_score_m2': 2000, 'lat': 48.1486, 'lon': 17.1077})
    df['location_score_m2'] = loc_stats['location_score_m2']
    df['lat'] = loc_stats['lat']
    df['lon'] = loc_stats['lon']

    cond_map = {
        'Novostavba': 4, 'Developersk\u00fd projekt': 4, 'Kompletn\u00e1 rekon\u0161trukcia': 3,
        '\u010ciasto\u010dn\u00e1 rekon\u0161trukcia': 2, 'P\u00f4vodn\u00fd stav': 1, 'Vo v\u00fdstavbe': 4, 'In\u00fd stav': 0,
    }
    df['condition_score'] = cond_map.get(data['stav_final'], 0)

    def haversine(lon1, lat1, lon2, lat2):
        from math import radians, cos, sin, asin, sqrt
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return c * 6371

    df['dist_bratislava'] = df.apply(lambda row: haversine(row['lon'], row['lat'], 17.1077, 48.1486), axis=1)

    row_lon, row_lat = loc_stats['lon'], loc_stats['lat']
    min_dist = float('inf')
    for city, (city_lat, city_lon) in DASHBOARD['REGIONAL_CITIES'].items():
        d = haversine(row_lon, row_lat, city_lon, city_lat)
        if d < min_dist:
            min_dist = d
    df['dist_nearest_city'] = min_dist

    df['log_area'] = np.log1p(data['floor_size'])
    df['log_land_area'] = np.log1p(data.get('land_area', 0))

    # Training data temporal features (scraped Feb 2026 — must match training)
    df['year_added'] = 2026
    df['month_added'] = 2

    # balkon/loggia COUNT features (separate from has_balcony/has_loggia binary)
    df['balkon'] = int(data.get('has_balcony', 0))
    df['loggia'] = int(data.get('has_loggia', 0))

    if category == 'byty':
        df['podlazie'] = data.get('current_floor', 0)

    df['relative_floor'] = df['current_floor'] / df['total_floors'] if data['total_floors'] > 0 else 0.5
    df['avg_room_size'] = df['floor_size'] / df['room_count'] if data['room_count'] > 0 else 0

    df['is_ground_floor'] = 1 if data.get('current_floor', 1) == 0 else 0
    df['is_top_floor'] = 1 if data.get('current_floor', 1) == data.get('total_floors', 1) else 0

    if category == 'byty':
        df['no_lift_high_floor'] = ((df['relative_floor'] > 0.5) & (df['has_lift'] == 0)).astype(int)
        df['built_up_ratio'] = 0
    else:
        df['built_up_ratio'] = df['built_up_area'] / df['land_area'] if data['land_area'] > 0 else 0
        df['no_lift_high_floor'] = 0

    df['utilities_score'] = sum([data['has_gas'], data['has_water'], data['has_electricity'], data['has_sewerage']])

    # Seasonality (use training data month — Feb 2026)
    df['month_sin'] = np.sin(2 * np.pi * 2 / 12)
    df['month_cos'] = np.cos(2 * np.pi * 2 / 12)
    df['days_on_market'] = 0

    # Binary amenity features
    df['has_loggia'] = int(data.get('has_loggia', 0))
    df['has_balcony'] = int(data.get('has_balcony', 0))
    df['has_cellar'] = int(data.get('has_cellar', 0))
    df['has_garage'] = int(data.get('has_garage', 0))
    df['has_parking'] = int(data.get('has_parking', 0))
    df['has_terrace'] = int(data.get('has_terrace', 0))
    df['has_pantry'] = int(data.get('has_pantry', 0))
    df['has_warehouse'] = int(data.get('has_warehouse', 0))
    df['has_ac'] = int(data.get('has_ac', 0))

    # One-Hot Encoding
    for col in feature_cols:
        if col not in df.columns:
            if col == f"stav_final_{data['stav_final']}":
                df[col] = 1
            elif col == f"construction_type_mapped_{data['construction']}":
                df[col] = 1
            elif col == f"heating_type_{data['heating']}":
                df[col] = 1
            elif col == f"vlastnictvo_{data.get('vlastnictvo', 'Unknown')}":
                df[col] = 1
            else:
                df[col] = 0

    df_final = df[feature_cols].copy()
    return df_final


# ============================================================
# MAIN APP — Apple HIG Single-Page Flow
# ============================================================

st.set_page_config(
    page_title="Odhad ceny nehnute\u013enosti | Slovensko",
    layout="wide",
    page_icon=None,
    menu_items={
        'About': 'Odhad ceny bytov a domov na Slovensku pomocou strojov\u00e9ho u\u010denia.'
    }
)

# --- Apple HIG CSS ---
st.markdown("""
<style>
/* ---- CSS Variables (Light mode) ---- */
:root {
    --bg-primary: linear-gradient(135deg, #f5f5f7, #e8e8ed);
    --surface-glass: rgba(255, 255, 255, 0.72);
    --surface-border: rgba(255, 255, 255, 0.5);
    --text-primary: #1d1d1f;
    --text-secondary: #86868b;
    --accent: #007aff;
    --positive: #34c759;
    --negative: #ff3b30;
    --warning: #ff9500;
}

/* ---- Dark mode variable architecture (toggle deferred) ---- */
[data-theme="dark"] {
    --bg-primary: linear-gradient(135deg, #000, #1c1c1e);
    --surface-glass: rgba(28, 28, 30, 0.72);
    --surface-border: rgba(255, 255, 255, 0.08);
    --text-primary: #f5f5f7;
    --text-secondary: #86868b;
    --accent: #0a84ff;
    --positive: #30d158;
    --negative: #ff453a;
    --warning: #ff9f0a;
}

/* ---- Global ---- */
.stApp {
    background: var(--bg-primary) !important;
}
.block-container {
    max-width: 900px !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* ---- Typography ---- */
h1 {
    font-family: -apple-system, 'SF Pro Display', system-ui, sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px !important;
    color: var(--text-primary) !important;
    text-align: center !important;
}
.subtitle {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: -1rem;
    margin-bottom: 1.5rem;
    font-family: -apple-system, 'SF Pro Text', system-ui, sans-serif;
}

/* ---- Glass Card via st.container(border=True) ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface-glass) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid var(--surface-border) !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
}
.card-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-secondary);
    font-weight: 500;
    margin-bottom: 0.5rem;
}

/* ---- Segment Toggle ---- */
div[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 0 !important;
    justify-content: center;
}
div[data-testid="stRadio"] > div > label {
    background: transparent;
    padding: 0.4rem 1.5rem !important;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    margin: 0 !important;
}
div[data-testid="stRadio"] > div > label[data-checked="true"],
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: white;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}
div[data-testid="stRadio"] > label { display: none !important; }

/* ---- Hero Price ---- */
.hero-price {
    text-align: center;
    font-family: -apple-system, 'SF Pro Display', system-ui, sans-serif;
    font-size: 3rem;
    font-weight: 600;
    letter-spacing: -1px;
    color: var(--text-primary);
    margin: 0.5rem 0;
}
.hero-subtitle {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
}

/* ---- Verdict Badge ---- */
.verdict-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}
.verdict-fair { background: rgba(52,199,89,0.12); color: #34c759; }
.verdict-over { background: rgba(255,59,48,0.12); color: #ff3b30; }
.verdict-slight-over { background: rgba(255,149,0,0.12); color: #ff9500; }
.verdict-under { background: rgba(255,149,0,0.12); color: #ff9500; }
.verdict-good { background: rgba(52,199,89,0.12); color: #34c759; }

/* ---- Stat Cards ---- */
.stat-card {
    background: var(--surface-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--surface-border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-card .stat-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}
.stat-card .stat-value {
    font-size: 1.2rem;
    font-weight: 500;
    color: var(--text-primary);
}

/* ---- Agreement Dots ---- */
.dot-green { color: #34c759; font-size: 1.1rem; }
.dot-yellow { color: #ff9500; font-size: 1.1rem; }
.dot-red { color: #ff3b30; font-size: 1.1rem; }

/* ---- Form Overrides ---- */
[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}
[data-testid="stFormSubmitButton"] button {
    background-color: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: opacity 0.2s ease;
}
[data-testid="stFormSubmitButton"] button:hover {
    opacity: 0.85;
}

/* ---- Expander as Glass ---- */
[data-testid="stExpander"] {
    background: var(--surface-glass) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--surface-border) !important;
    border-radius: 16px !important;
    margin-bottom: 1rem;
}

/* ---- Footer ---- */
.app-footer {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
    line-height: 1.6;
}
.app-footer a {
    color: var(--accent);
    text-decoration: none;
}
.app-footer a:hover {
    text-decoration: underline;
}

/* ---- Hide sidebar ---- */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("Predikcia Ceny Nehnute\u013enost\u00ed")
st.markdown('<p class="subtitle">Ensemble ML model \u00b7 10 000+ nehnute\u013enost\u00ed</p>', unsafe_allow_html=True)

# --- Session State ---
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'input_data' not in st.session_state:
    st.session_state.input_data = None

# --- Help Section ---
with st.expander("Ako to funguje?"):
    st.markdown("""
**Tento n\u00e1stroj odhaduje trhov\u00fa cenu nehnute\u013enosti** na z\u00e1klade podobn\u00fdch inzer\u00e1tov z nehnutelnosti.sk.

- **Interval neistoty** \u2014 rozsah, v ktorom sa re\u00e1lna cena pravdepodobne nach\u00e1dza
- **Zhoda modelov** \u2014 nako\u013eko sa 4 nez\u00e1visl\u00e9 modely zhoduj\u00fa (vy\u0161\u0161ia zhoda = spo\u013eahliv\u0161\u00ed odhad)
- **SHAP anal\u00fdza** \u2014 vysvet\u013euje, ktor\u00e9 faktory cenu najviac ovplyv\u0148uj\u00fa a o ko\u013eko EUR

*Odhad sl\u00fa\u017ei ako orient\u00e1cia, nie ako znaleck\u00fd posudok.*
""")

# --- Segment Toggle (outside form) ---
category = st.radio("Segment", ["byty", "domy"], horizontal=True, label_visibility="collapsed")

# --- Load Models ---
with st.spinner("Na\u010d\u00edtavam modely... Pros\u00edm \u010dakajte."):
    mappings, features, models = load_resources(category)


# ============================================================
# SECTION 1: INPUT FORM \u2014 2x2 Glass Cards
# ============================================================

with st.form("input_form"):
    row1_col1, row1_col2 = st.columns(2)

    # --- Card 1: Nehnute\u013enos\u0165 ---
    with row1_col1:
        with st.container(border=True):
            st.markdown('<p class="card-label">Nehnute\u013enos\u0165</p>', unsafe_allow_html=True)
            floor_size = st.number_input(
                "Plocha (m\u00b2)", min_value=10, max_value=500, value=60,
                help="\u00da\u017eitkov\u00e1 plocha nehnute\u013enosti."
            )
            room_count = st.number_input(
                "Po\u010det izieb", min_value=1, max_value=10, value=2,
            )
            stav = st.selectbox("Stav", mappings['options']['stav_final'])
            construction = st.selectbox("Kon\u0161trukcia", mappings['options']['construction'])
            if category == 'byty':
                current_floor = st.number_input("Poschodie", min_value=0, max_value=30, value=2)
                total_floors = st.number_input("Po\u010det poschod\u00ed v budove", min_value=1, max_value=30, value=5)
                has_lift = st.toggle("V\u00fd\u0165ah", value=False)
                land_area = 0
                built_up_area = 0
            else:
                current_floor = 0
                total_floors = st.number_input("Po\u010det podla\u017e\u00ed", min_value=1, max_value=5, value=1)
                land_area = st.number_input("Plocha pozemku (m\u00b2)", min_value=0, max_value=5000, value=400)
                built_up_area = st.number_input("Zastavan\u00e1 plocha (m\u00b2)", min_value=0, max_value=500, value=100)
                has_lift = False

    # --- Card 2: Lokalita + Ostatn\u00e9 ---
    with row1_col2:
        with st.container(border=True):
            st.markdown('<p class="card-label">Lokalita</p>', unsafe_allow_html=True)
            all_locations = sorted(mappings['locations'].keys())
            location = st.selectbox(
                "Obec / mestská časť", all_locations,
                help="Začnite písať pre filtrovanie."
            )

            vlastnictvo = st.selectbox("Vlastn\u00edctvo",
                ['Osobn\u00e9', 'Dru\u017estevn\u00e9', 'Firemn\u00e9', 'Obecn\u00e9', '\u0160t\u00e1tne', 'In\u00e9', 'Nevysporiadan\u00e9', 'Unknown'])
            heating = st.selectbox("K\u00farenie", mappings['options']['heating'])

    row2_col1, row2_col2 = st.columns(2)

    # --- Card 3: Vybavenie ---
    with row2_col1:
        with st.container(border=True):
            st.markdown('<p class="card-label">Vybavenie</p>', unsafe_allow_html=True)
            vybavenie_cols = st.columns(2)
            with vybavenie_cols[0]:
                has_balcony = st.toggle("Balk\u00f3n", value=False)
                has_loggia = st.toggle("Loggia", value=False)
                has_cellar = st.toggle("Pivnica", value=False)
            with vybavenie_cols[1]:
                has_garage = st.toggle("Gar\u00e1\u017e", value=False)
                has_parking = st.toggle("Parkovanie", value=False)
                has_terrace = st.toggle("Terasa", value=False)

    # --- Card 4: Porovnanie s inzer\u00e1tom ---
    with row2_col2:
        with st.container(border=True):
            st.markdown('<p class="card-label">Porovnanie s inzer\u00e1tom</p>', unsafe_allow_html=True)
            market_price = st.number_input(
                "Inzerovan\u00e1 cena (\u20ac)", min_value=0, value=0,
                help="Volite\u013en\u00e9 \u2014 zadajte cenu z inzer\u00e1tu pre porovnanie s odhadom."
            )

    submitted = st.form_submit_button("Odhadn\u00fa\u0165 cenu", type="primary")

if submitted:
    st.session_state.prediction_done = True
    st.session_state.input_data = {
        'floor_size': floor_size,
        'obec_cast': location,
        'stav_final': stav,
        'construction': construction,
        'room_count': room_count,
        'current_floor': current_floor,
        'total_floors': total_floors,
        'has_lift': int(has_lift),
        'has_balcony': int(has_balcony),
        'has_loggia': int(has_loggia),
        'has_cellar': int(has_cellar),
        'has_garage': int(has_garage),
        'has_parking': int(has_parking),
        'has_terrace': int(has_terrace),
        'land_area': land_area,
        'built_up_area': built_up_area,
        'has_gas': 1,
        'has_water': 1,
        'has_electricity': 1,
        'has_sewerage': 1,
        'heating': heating,
        'vlastnictvo': vlastnictvo,
        'market_price': market_price,
    }


# ============================================================
# SECTIONS 2-6: Visible only after prediction
# ============================================================

if st.session_state.prediction_done and st.session_state.input_data:
    input_data = st.session_state.input_data
    market_price = input_data.get('market_price', 0)

    # --- Prediction ---
    X_input = process_input(input_data, mappings, features, category)
    final_price, (pred_xgb, pred_lgb, pred_cb, pred_nn) = predict_ensemble(X_input, models)
    ci_lower, ci_upper = calculate_confidence_interval(final_price, category)
    model_prices = [np.exp(pred_xgb)[0], np.exp(pred_lgb)[0], np.exp(pred_cb)[0], np.exp(pred_nn)[0]]

    # SHAP (computed once, reused across sections — always on)
    shap_values = None
    with st.spinner("Po\u010d\u00edtam vysvetlenie modelu..."):
        explainer = get_shap_explainer(models['xgb'], f'xgb_{category}')
        shap_values = explainer(X_input)

    # ========================================================
    # SECTION 2: HERO PRICE
    # ========================================================

    st.markdown(f'<div class="hero-price">{final_price:,.0f} \u20ac</div>', unsafe_allow_html=True)

    # Verdict badge (only if market price provided)
    if market_price > 0:
        diff = market_price - final_price
        diff_pct = (diff / final_price) * 100
        if abs(diff_pct) < 5:
            badge_class, badge_text = "verdict-fair", f"F\u00e9rov\u00e1 cena ({diff_pct:+.1f}%)"
        elif diff_pct > 20:
            badge_class, badge_text = "verdict-over", f"V\u00fdrazne nadhodnoten\u00e1 (+{diff_pct:.1f}%)"
        elif diff_pct > 5:
            badge_class, badge_text = "verdict-slight-over", f"Mierne nadhodnoten\u00e1 (+{diff_pct:.1f}%)"
        elif diff_pct < -20:
            badge_class, badge_text = "verdict-under", f"V\u00fdrazne podhodnoten\u00e1 ({diff_pct:.1f}%)"
        else:
            badge_class, badge_text = "verdict-good", f"V\u00fdhodn\u00e1 k\u00fapa ({diff_pct:.1f}%)"
        st.markdown(
            f'<div style="text-align:center"><span class="verdict-badge {badge_class}">{badge_text}</span></div>',
            unsafe_allow_html=True
        )

    # Price context subtitle
    price_per_m2 = final_price / input_data['floor_size']
    loc_stats_hero = mappings['locations'].get(input_data['obec_cast'], {'location_score_m2': 0})
    loc_avg_m2 = loc_stats_hero['location_score_m2']
    st.markdown(
        f'<p class="hero-subtitle">{price_per_m2:,.0f} \u20ac/m\u00b2 \u00b7 priemer lokality {loc_avg_m2:,.0f} \u20ac/m\u00b2</p>',
        unsafe_allow_html=True
    )

    # --- Three stat cards: Interval | \u20ac/m\u00b2 | Zhoda ---
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-label">Interval neistoty</div>
            <div class="stat-value">{ci_lower:,.0f} \u20ac \u2014 {ci_upper:,.0f} \u20ac</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-label">Cena za m\u00b2</div>
            <div class="stat-value">{price_per_m2:,.0f} \u20ac/m\u00b2</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        cv = np.std(model_prices) / np.mean(model_prices) if np.mean(model_prices) > 0 else 0
        if cv < 0.05:
            dot_html, agreement_label = '<span class="dot-green">\u25cf</span>', "Vysok\u00e1 zhoda"
        elif cv < 0.15:
            dot_html, agreement_label = '<span class="dot-yellow">\u25cf</span>', "Mierna zhoda"
        else:
            dot_html, agreement_label = '<span class="dot-red">\u25cf</span>', "N\u00edzka zhoda"
        st.markdown(f"""<div class="stat-card">
            <div class="stat-label">Zhoda modelov</div>
            <div class="stat-value">{dot_html} {agreement_label} ({cv:.1%})</div>
        </div>""", unsafe_allow_html=True)

    # ========================================================
    # SECTION 3: MAP + SHAP
    # ========================================================

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    map_col, shap_col = st.columns([1, 1])

    with map_col:
        with st.container(border=True):
            st.markdown('<p class="card-label">Poloha nehnute\u013enosti</p>', unsafe_allow_html=True)
            loc_name = input_data['obec_cast']
            loc_viz = mappings['locations'].get(loc_name, {'lat': 48.1486, 'lon': 17.1077})
            map_data = pd.DataFrame({
                'lat': [loc_viz['lat']], 'lon': [loc_viz['lon']],
                'name': [loc_name], 'price': [f"{final_price:,.0f}"],
                'area': [input_data['floor_size']]
            })
            st.pydeck_chart(pdk.Deck(
                map_style=MAP_STYLES["Light"],
                initial_view_state=pdk.ViewState(
                    latitude=loc_viz['lat'], longitude=loc_viz['lon'],
                    zoom=13, pitch=45),
                layers=[pdk.Layer("ScatterplotLayer", data=map_data,
                    get_position='[lon, lat]', get_color='[200, 30, 0, 160]',
                    get_radius=200, pickable=True)],
                tooltip={"text": "{name}\nCena: {price} \u20ac\nPlocha: {area} m\u00b2"}
            ), use_container_width=True)

    with shap_col:
        with st.container(border=True):
            st.markdown('<p class="card-label">\u010co najviac ovplyv\u0148uje cenu?</p>', unsafe_allow_html=True)
            if shap_values is not None:
                top_factors = get_plain_language_shap(shap_values[0], final_price, X_input=X_input, top_n=5)
                for label, eur, fmt_val in top_factors:
                    direction = "zvy\u0161uje" if eur > 0 else "zni\u017euje"
                    icon = "\U0001f53a" if eur > 0 else "\U0001f53b"
                    val_str = f" ({fmt_val})" if fmt_val else ""
                    st.markdown(f"{icon} **{label}**{val_str} {direction} cenu o **{abs(eur):,.0f} \u20ac**")
                st.caption("Na z\u00e1klade XGBoost SHAP anal\u00fdzy. Hodnoty s\u00fa aproxim\u00e1cie vplyvu na fin\u00e1lnu cenu.")

    # ========================================================
    # SECTION 4: WHAT-IF ANALYSIS
    # ========================================================

    with st.container(border=True):
        st.markdown('<p class="card-label">What-If Anal\u00fdza</p>', unsafe_allow_html=True)

        # 4a: Renovation simulation
        st.markdown("#### Simul\u00e1cia rekon\u0161trukcie")
        current_stav = input_data['stav_final']
        if current_stav not in ('Kompletn\u00e1 rekon\u0161trukcia', 'Novostavba', 'Developersk\u00fd projekt'):
            input_renovated = input_data.copy()
            input_renovated['stav_final'] = 'Kompletn\u00e1 rekon\u0161trukcia'
            X_renovated = process_input(input_renovated, mappings, features, category)
            price_renovated, _ = predict_ensemble(X_renovated, models)
            increase = price_renovated - final_price
            st.metric("Potenci\u00e1lna cena po rekon\u0161trukcii", f"{price_renovated:,.0f} \u20ac", delta=f"+{increase:,.0f} \u20ac")
        else:
            st.info("Nehnute\u013enos\u0165 je u\u017e v top stave (Novostavba / Rekon\u0161trukcia).")

        # 4b: Location comparison
        st.markdown("---")
        st.markdown("#### Porovnanie lokal\u00edt")

        all_locations = sorted(mappings['locations'].keys())
        whatif_search = st.text_input("Hľadať lokalitu...", value="", key="whatif_search",
                                       help="Funguje aj bez diakritiky (napr. 'Presov').")
        if whatif_search.strip():
            whatif_query = strip_diacritics(whatif_search).lower()
            whatif_options = [loc for loc in all_locations
                             if loc != input_data['obec_cast'] and whatif_query in strip_diacritics(loc).lower()]
        else:
            whatif_options = [loc for loc in all_locations if loc != input_data['obec_cast']]
        target_locs = st.multiselect(
            "Porovnajte cenu v iných lokalitách (max. 3):",
            whatif_options,
            max_selections=3,
            key="whatif_locations"
        )

        if target_locs:
            comparison_results = []
            for loc in target_locs:
                input_loc = input_data.copy()
                input_loc['obec_cast'] = loc
                X_loc = process_input(input_loc, mappings, features, category)
                price_loc, _ = predict_ensemble(X_loc, models)
                loc_stats_comp = mappings['locations'].get(loc, {'location_score_m2': 0, 'lat': 48.15, 'lon': 17.11})
                shap_diff_top = []
                if shap_values is not None:
                    shap_loc = get_shap_explainer(models['xgb'], f'xgb_{category}')(X_loc)
                    shap_diff_top = get_shap_diff_top(shap_values[0], shap_loc[0], price_loc)
                comparison_results.append({
                    'location': loc, 'price': price_loc,
                    'price_m2': price_loc / input_data['floor_size'],
                    'loc_avg_m2': loc_stats_comp['location_score_m2'],
                    'lat': loc_stats_comp['lat'], 'lon': loc_stats_comp['lon'],
                    'shap_diff': shap_diff_top,
                })

            n_cols = 1 + len(comparison_results)
            cols = st.columns(n_cols)
            with cols[0]:
                st.markdown(f"**{input_data['obec_cast']}** (origin\u00e1l)")
                st.metric("Cena", f"{final_price:,.0f} \u20ac")
                st.caption(f"{price_per_m2:,.0f} \u20ac/m\u00b2")

            for i, comp in enumerate(comparison_results):
                with cols[i + 1]:
                    delta = comp['price'] - final_price
                    st.markdown(f"**{comp['location']}**")
                    st.metric("Cena", f"{comp['price']:,.0f} \u20ac", delta=f"{delta:+,.0f} \u20ac")
                    st.caption(f"{comp['price_m2']:,.0f} \u20ac/m\u00b2")
                    if comp['shap_diff']:
                        for lbl, eur_d in comp['shap_diff']:
                            icon = "\U0001f53a" if eur_d > 0 else "\U0001f53b"
                            st.markdown(f"{icon} {lbl}: **{eur_d:+,.0f} \u20ac**")

            # Multi-pin map
            loc_orig = mappings['locations'].get(input_data['obec_cast'], {'lat': 48.15, 'lon': 17.11})
            pin_colors = [[200, 30, 0, 180], [30, 130, 200, 180], [30, 180, 30, 180], [255, 150, 0, 180]]
            all_pins = [{
                'lat': loc_orig['lat'], 'lon': loc_orig['lon'],
                'name': input_data['obec_cast'] + ' (origin\u00e1l)',
                'price': f"{final_price:,.0f}", 'color': pin_colors[0],
            }]
            for i, comp in enumerate(comparison_results):
                all_pins.append({
                    'lat': comp['lat'], 'lon': comp['lon'],
                    'name': comp['location'],
                    'price': f"{comp['price']:,.0f}", 'color': pin_colors[i + 1],
                })
            pins_df = pd.DataFrame(all_pins)
            avg_lat = pins_df['lat'].mean()
            avg_lon = pins_df['lon'].mean()
            lat_range = pins_df['lat'].max() - pins_df['lat'].min()
            lon_range = pins_df['lon'].max() - pins_df['lon'].min()
            max_range = max(lat_range, lon_range, 0.01)
            zoom = max(5, min(13, 8 - np.log2(max_range + 0.001)))
            st.pydeck_chart(pdk.Deck(
                map_style=MAP_STYLES["Light"],
                initial_view_state=pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=zoom, pitch=30),
                layers=[pdk.Layer("ScatterplotLayer", data=pins_df,
                    get_position='[lon, lat]', get_color='color',
                    get_radius=300, pickable=True)],
                tooltip={"text": "{name}\nCena: {price} \u20ac"}
            ), use_container_width=True)

    # ========================================================
    # SECTION 5: DETAILED ANALYSIS (collapsed)
    # ========================================================

    with st.expander("Podrobn\u00e1 anal\u00fdza"):
        st.markdown("#### SHAP Waterfall")
        shap_model_name = st.selectbox(
            "Model pre vysvetlenie:",
            ["XGBoost", "LightGBM", "CatBoost"],
            key="shap_model_detail",
        )
        model_key_map = {"XGBoost": 'xgb', "LightGBM": 'lgb', "CatBoost": 'cb'}
        detail_key = model_key_map[shap_model_name]
        detail_explainer = get_shap_explainer(models[detail_key], f'{detail_key}_{category}')
        detail_shap = detail_explainer(X_input)
        shap.plots.waterfall(detail_shap[0], show=False, max_display=10)
        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        st.pyplot(fig, use_container_width=False)
        plt.clf()
        st.caption("**\u010cerven\u00e9 (+):** zvy\u0161uj\u00fa cenu oproti priemeru. **Modr\u00e9 (-):** zni\u017euj\u00fa cenu oproti priemeru.")

        st.markdown("#### Porovnanie modelov")
        details = pd.DataFrame({
            'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'Neural Network', 'Ensemble (Final)'],
            'Predikcia (\u20ac)': model_prices + [final_price],
            'Odch\u00fdlka': [f"{((p - final_price) / final_price) * 100:+.1f}%" for p in model_prices] + ['\u2014']
        })
        st.table(details.style.format({'Predikcia (\u20ac)': '{:,.0f}'}))

        median_price = np.median(model_prices)
        clipped = [min(max(p, median_price * 0.3), median_price * 3.0) for p in model_prices]
        chart_df = pd.DataFrame({
            'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'Neural Network', 'Ensemble'],
            'Predikcia (\u20ac)': clipped + [final_price]
        })
        st.bar_chart(chart_df.set_index('Model'))

    # ========================================================
    # SECTION 6: TECHNICAL DETAILS (collapsed)
    # ========================================================

    with st.expander("Technick\u00e9 detaily modelu"):
        mape_display = DASHBOARD['MAPE_BYTY'] if category == 'byty' else DASHBOARD['MAPE_DOMY']
        st.markdown(f"""
**Architekt\u00fara:** Stacking Ensemble (XGB + LGB + CB + NN \u2192 Ridge)

**Dataset:** {'10,773 bytov' if category == 'byty' else '11,060 domov'} (po filtr\u00e1cii anom\u00e1li\u00ed)

**Po\u010det atrib\u00fatov:** {'73' if category == 'byty' else '72'}

**Valida\u010dn\u00e9 metriky ({category}):**
| Metrika | Hodnota |
|---|---|
| MAE | {'27,388 \u20ac' if category == 'byty' else '55,639 \u20ac'} |
| MAPE | {mape_display*100:.1f}% |
| R\u00b2 | {'0.856' if category == 'byty' else '0.739'} |

**Feature Selection:** RFECV {'30' if category == 'byty' else '37'}/{'73' if category == 'byty' else '72'} \u2192 <0.6% MAE n\u00e1rast

**Tr\u00e9ning:** Febru\u00e1r 2026
""")

    # Store results in session state
    st.session_state.prediction_results = {
        'final_price': final_price, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'X_input': X_input, 'pred_xgb': pred_xgb, 'pred_lgb': pred_lgb,
        'pred_cb': pred_cb, 'pred_nn': pred_nn, 'model_prices': model_prices,
    }

# ========================================================
# FOOTER
# ========================================================
st.markdown("""
<div class="app-footer">
    <strong>Proof of Concept</strong> \u00b7 Experiment\u00e1lny n\u00e1stroj na odhad cien nehnute\u013enost\u00ed<br>
    Model tr\u00e9novan\u00fd na 10 000+ inzer\u00e1toch z nehnutelnosti.sk \u00b7 XGBoost + LightGBM + CatBoost + NN \u2192 Ensemble<br>
    Presnos\u0165: MAPE 13.1% (byty), 26.9% (domy) \u00b7 Odhad sl\u00fa\u017ei ako orient\u00e1cia, nie ako znaleck\u00fd posudok<br><br>
    <a href="https://github.com/adamJesensky/slovak-real-estate-predictor/issues">Nahl\u00e1si\u0165 chybu</a>
    &nbsp;\u00b7&nbsp;
    <a href="mailto:ajesensky8@gmail.com">Sp\u00e4tn\u00e1 v\u00e4zba</a>
</div>
""", unsafe_allow_html=True)
