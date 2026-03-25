
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
from streamlit_searchbox import st_searchbox

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ASSETS_DIR = os.path.join(BASE_DIR, 'dashboard', 'assets')

sys.path.insert(0, os.path.join(BASE_DIR))
sys.path.insert(0, os.path.join(BASE_DIR, 'dashboard'))
from config import DASHBOARD
from listing_scraper import fetch_listing, build_model_input as _build_scraper_input

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
def _load_model_set(tag):
    """Load a complete model set (XGB + LGB + CB + NN + meta) for a given tag."""
    features = joblib.load(os.path.join(MODELS_DIR, f'features_{tag}.joblib'))
    models = {}
    models['xgb'] = joblib.load(os.path.join(MODELS_DIR, f'xgb_{tag}.joblib'))
    models['lgb'] = joblib.load(os.path.join(MODELS_DIR, f'lgb_{tag}.joblib'))
    cb = CatBoostRegressor()
    cb.load_model(os.path.join(MODELS_DIR, f'cb_{tag}.cbm'))
    models['cb'] = cb
    models['meta'] = joblib.load(os.path.join(MODELS_DIR, f'meta_{tag}.joblib'))
    imputer = joblib.load(os.path.join(MODELS_DIR, f'nn_imputer_{tag}.joblib'))
    scaler = joblib.load(os.path.join(MODELS_DIR, f'nn_scaler_{tag}.joblib'))
    device = torch.device('cpu')
    nn_model = AdvancedNN(len(features))
    nn_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'nn_{tag}.pth'), map_location=device))
    nn_model.eval()
    models['nn'] = nn_model
    models['nn_imputer'] = imputer
    models['nn_scaler'] = scaler
    return features, models

@st.cache_resource
def load_resources(category):
    with open(os.path.join(ASSETS_DIR, f'mappings_{category}.json'), 'r', encoding='utf-8') as f:
        mappings = json.load(f)

    if category == 'byty':
        # Regional models: separate BA and non-BA model sets
        features_ba, models_ba = _load_model_set('byty_ba')
        features_nonba, models_nonba = _load_model_set('byty_nonba')
        model_sets = {
            'ba': (features_ba, models_ba),
            'nonba': (features_nonba, models_nonba),
        }
    else:
        features, models = _load_model_set('domy')
        model_sets = {'all': (features, models)}

    return mappings, model_sets

def get_region(category, location_name):
    """Determine which model region to use based on location."""
    if category == 'byty' and location_name and location_name.startswith('Bratislava'):
        return 'ba'
    elif category == 'byty':
        return 'nonba'
    return 'all'

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
    # Anchor NN to tree model consensus — if NN diverges too far, snap to median
    tree_median = np.median([pred_xgb[0], pred_lgb[0], pred_cb[0]])
    nn_divergence = abs(pred_nn[0] - tree_median)
    if nn_divergence > 0.5:  # >0.5 in log space ≈ 65% price difference
        pred_nn = np.array([tree_median])
    stack_X = np.column_stack((pred_xgb, pred_lgb, pred_cb, pred_nn))
    final_log = models['meta'].predict(stack_X)
    final_price = np.exp(final_log)[0]
    return final_price, (pred_xgb, pred_lgb, pred_cb, pred_nn)

FEATURE_LABELS = {
    'location_score_m2': 'Lokalita (cena/m²)',
    'floor_size': 'Plocha',
    'log_area': 'Plocha',
    'condition_score': 'Stav nehnuteľnosti',
    'dist_bratislava': 'Vzdialenosť od Bratislavy',
    'dist_nearest_city': 'Vzdialenosť od krajského mesta',
    'room_count': 'Počet izieb',
    'has_lift': 'Výťah',
    'relative_floor': 'Relatívne poschodie',
    'utilities_score': 'Inž. siete (N/A)',
    'has_balcony': 'Balkón',
    'has_loggia': 'Loggia',
    'has_cellar': 'Pivnica',
    'has_garage': 'Garáž',
    'has_parking': 'Parkovanie',
    'has_terrace': 'Terasa',
    'land_area': 'Plocha pozemku',
    'log_land_area': 'Plocha pozemku',
    'built_up_area': 'Zastavaná plocha',
    'built_up_ratio': 'Pomer zastavanej plochy',
    'avg_room_size': 'Priemerná veľkosť izby',
    'current_floor': 'Poschodie',
    'total_floors': 'Počet poschodí',
    'no_lift_high_floor': 'Vysoké poschodie bez výťahu',
    'is_ground_floor': 'Prízemie',
    'is_top_floor': 'Posledné poschodie',
    'balkon': 'Balkón (počet)',
    'loggia': 'Loggia (počet)',
    'podlazie': 'Podlažie',
    'days_on_market': 'Dni na trhu',
    'has_gas': 'Plyn',
    'has_water': 'Voda',
    'has_electricity': 'Elektrina',
    'has_sewerage': 'Kanalizácia',
    'has_pantry': 'Špajza',
    'has_warehouse': 'Sklad',
    'has_ac': 'Klimatizácia',
    'month_sin': 'Sezónnosť',
    'month_cos': 'Sezónnosť',
    'year_added': 'Rok inzerátu',
    'month_added': 'Mesiac inzerátu',
}

def shap_to_eur(shap_value, final_price):
    """Convert SHAP value (ln-price space) to EUR impact."""
    return final_price * (1 - np.exp(-shap_value))

def _format_feature_value(fname, raw_value):
    """Format a feature's raw value for display next to SHAP explanation."""
    if fname in ('location_score_m2',):
        return f"{raw_value:,.0f} €/m²"
    if fname in ('floor_size', 'land_area', 'built_up_area'):
        return f"{raw_value:,.0f} m²"
    if fname in ('log_area', 'log_land_area'):
        return f"{np.expm1(raw_value):,.0f} m²"
    if fname in ('dist_bratislava', 'dist_nearest_city'):
        return f"{raw_value:,.0f} km"
    if fname in ('room_count', 'current_floor', 'total_floors', 'balkon', 'loggia', 'podlazie'):
        return f"{int(raw_value)}"
    if fname == 'condition_score':
        score_labels = {0: 'Iný', 1: 'Pôvodný', 2: 'Čiastočná rek.', 3: 'Kompletná rek.', 4: 'Novostavba'}
        return score_labels.get(int(raw_value), str(int(raw_value)))
    if fname == 'relative_floor':
        return f"{raw_value:.0%}"
    if fname == 'avg_room_size':
        return f"{raw_value:,.0f} m²"
    if fname in ('days_on_market',):
        return f"{int(raw_value)} dní"
    if fname == 'built_up_ratio':
        return f"{raw_value:.0%}"
    # Binary features
    if fname.startswith('has_') or fname in ('is_ground_floor', 'is_top_floor', 'no_lift_high_floor'):
        return "áno" if raw_value >= 0.5 else "nie"
    # One-hot encoded categories — extract the category name
    for prefix in ('stav_final_', 'construction_type_mapped_', 'heating_type_', 'vlastnictvo_'):
        if fname.startswith(prefix):
            return "áno" if raw_value >= 0.5 else "nie"
    return None

def get_plain_language_shap(shap_values_row, final_price, X_input=None, top_n=5):
    """Return list of (feature_label, eur_impact, formatted_value) sorted by |impact|."""
    values = shap_values_row.values
    feature_names = shap_values_row.feature_names
    impacts = []
    for sv, fname in zip(values, feature_names):
        eur = shap_to_eur(sv, final_price)
        label = translate_feature_name(fname)
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
        lbl = translate_feature_name(fn)
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

# --- Slovak UI Labels for Dropdowns ---
CONSTRUCTION_LABELS = {
    'Brick': 'Tehla', 'Block': 'Blok / tvárnice', 'Mixed': 'Zmiešaná',
    'Other': 'Iná', 'Panel': 'Panel', 'ReinforcedConcrete': 'Železobetón',
    'Stone': 'Kameň', 'Wood': 'Drevo', 'Skeleton': 'Skelet',
    'Prefab': 'Prefabrikát', 'Unknown': 'Neznáma',
}
HEATING_LABELS = {
    'central': 'Ústredné', 'local': 'Lokálne', 'other': 'Iné',
    'underfloor': 'Podlahové', 'unknown': 'Neznáme',
}

def translate_feature_name(fname):
    """Translate a model feature name to Slovak for SHAP display."""
    if fname in FEATURE_LABELS:
        return FEATURE_LABELS[fname]
    for prefix, label, value_map in [
        ('stav_final_', 'Stav', None),
        ('construction_type_mapped_', 'Konštrukcia', CONSTRUCTION_LABELS),
        ('heating_type_', 'Kúrenie', HEATING_LABELS),
        ('vlastnictvo_', 'Vlastníctvo', None),
    ]:
        if fname.startswith(prefix):
            value = fname[len(prefix):]
            if value_map:
                value = value_map.get(value, value)
            return f"{label}: {value}"
    return fname

def strip_diacritics(text):
    """Remove diacritics: 'Žilina' -> 'Zilina', 'Prešov' -> 'Presov'."""
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
        'Novostavba': 4, 'Developerský projekt': 4, 'Kompletná rekonštrukcia': 3,
        'Čiastočná rekonštrukcia': 2, 'Pôvodný stav': 1, 'Vo výstavbe': 4,
        'Určený k demolácii': 0, 'undefined': 0,
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
    page_title="Odhad ceny nehnuteľnosti | Slovensko",
    layout="wide",
    page_icon=None,
    menu_items={
        'About': 'Odhad ceny bytov a domov na Slovensku pomocou strojového učenia.'
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
.verdict-fair { background: rgba(52,199,89,0.15); color: #248a3d; }
.verdict-over { background: rgba(255,59,48,0.15); color: #d70015; }
.verdict-slight-over { background: rgba(255,149,0,0.15); color: #b25000; }
.verdict-under { background: rgba(255,149,0,0.15); color: #b25000; }
.verdict-good { background: rgba(52,199,89,0.15); color: #248a3d; }

/* ---- Stat Cards ---- */
.stat-card {
    background: var(--surface-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--surface-border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    min-height: 5.5rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.stat-card .stat-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}
.stat-card .stat-value {
    font-size: 1.1rem;
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
}

/* ---- Scraper Banner ---- */
.scraper-banner {
    background: var(--surface-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--surface-border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.scraper-banner .scraper-title {
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
}
.scraper-banner .scraper-price {
    font-size: 0.85rem;
    color: var(--accent);
    font-weight: 500;
}
.scraper-warning-item {
    font-size: 0.8rem;
    color: var(--warning);
    padding: 0.15rem 0;
}
.scraper-banner .scraper-hint {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
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

/* ---- Inputs: rounded corners to match glass cards ---- */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div,
.stSearchbox input,
.stSearchbox > div > div > div {
    border-radius: 10px !important;
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

/* ---- Responsive: Tablets (≤ 1024px) ---- */
@media (max-width: 1024px) {
    .block-container {
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}

/* ---- Responsive: Mobile (≤ 640px) ---- */
@media (max-width: 640px) {
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    h1 {
        font-size: 1.5rem !important;
    }
    .hero-price {
        font-size: 2rem;
    }
    .hero-subtitle {
        font-size: 0.78rem;
    }
    .verdict-badge {
        font-size: 0.78rem;
        padding: 0.25rem 0.8rem;
    }
    .stat-card {
        min-height: auto;
        padding: 0.8rem 1rem;
        margin-bottom: 0.3rem;
    }
    .stat-card .stat-value {
        font-size: 0.95rem;
        white-space: normal;
    }
    .stat-card .stat-label {
        font-size: 0.65rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        padding: 0.3rem !important;
    }
    .card-label {
        font-size: 0.65rem;
    }
    .scraper-banner {
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stRadio"] > div > label {
        padding: 0.35rem 1rem !important;
        font-size: 0.85rem;
    }
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("Predikcia Ceny Nehnuteľností")
st.markdown('<p class="subtitle">Odhad na základe 13 000+ reálnych inzerátov</p>', unsafe_allow_html=True)

# --- Session State ---
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'scraper_result' not in st.session_state:
    st.session_state.scraper_result = None

# --- Help Section ---
with st.expander("Ako to funguje?"):
    st.markdown("""
**Tento nástroj odhaduje trhovú cenu nehnuteľnosti** na základe 13 000+ reálnych inzerátov z nehnutelnosti.sk (február 2026).

**Čo znamenajú výsledky:**
- **Odhadovaná cena** — najpravdepodobnejšia trhová cena na základe zadaných parametrov
- **Interval neistoty** — rozsah, v ktorom sa reálna cena s veľkou pravdepodobnosťou nachádza. Čím užší interval, tým istejší odhad.
- **Zhoda modelov** — 4 nezávislé modely odhadujú cenu každý zvlášť. Ak sa ich odhady líšia málo (zelená bodka), výsledok je spoľahlivejší. Väčšie rozdiely (žltá/červená) naznačujú menej istý odhad.
- **Čo ovplyvňuje cenu** — zobrazuje, ktoré faktory cenu najviac zvyšujú alebo znižujú oproti priemeru, a o koľko EUR

**Obmedzenia:**
- Odhad vychádza z inzerovaných cien, nie z reálnych predajných cien
- Model nezohľadňuje vnútornú dispozíciu, výhľad, hluk ani stav interiéru
- Pre lokality s málo inzerátmi môže byť odhad menej presný
- Byty v Bratislave a mimo Bratislavy používajú oddelené modely optimalizované pre každý trh

*Odhad slúži ako orientácia, nie ako znalecký posudok.*
""")

# --- Segment Toggle (outside form) ---
_cat_label = st.radio("Segment", ["Byty", "Domy"], horizontal=True,
                       label_visibility="collapsed", key="segment_toggle")
category = _cat_label.lower()

# --- Load Models ---
with st.spinner("Načítavam modely... Prosím čakajte."):
    mappings, model_sets = load_resources(category)

def _apply_scraper_result(inp, warns, conf, title):
    """Store scraper output in session_state, bump form/searchbox versions, and rerun."""
    st.session_state.scraper_result = {
        'input': inp, 'warnings': warns, 'confidence': conf, 'title': title,
    }
    # Remove any stale amenity keys from session state — the toggle widgets will
    # re-create them from the `value=` parameter (via _pf) on the next render.
    # Setting both session_state[key] AND value= causes a Streamlit conflict.
    for _ak in ['has_balcony', 'has_loggia', 'has_cellar', 'has_garage', 'has_parking',
                 'has_terrace', 'has_pantry', 'has_warehouse', 'has_ac']:
        st.session_state.pop(f'am_{_ak}', None)
    st.session_state._form_ver = st.session_state.get('_form_ver', 0) + 1
    st.session_state._sb_ver = st.session_state.get('_sb_ver', 0) + 1
    st.session_state.prediction_done = False
    st.session_state.input_data = None

# --- Process pending scraper result (from category switch) ---
if '_scraper_pending' in st.session_state:
    _pending = st.session_state.pop('_scraper_pending')
    _inp, _warns, _conf = _build_scraper_input(_pending['parsed'], category, mappings)
    _apply_scraper_result(_inp, _warns, _conf, _pending.get('title', ''))
    st.rerun()

# --- URL Scraper Section ---
with st.container(border=True):
    st.markdown('<p class="card-label">Načítať z inzerátu</p>', unsafe_allow_html=True)
    _url_col, _btn_col = st.columns([5, 1])
    with _url_col:
        _listing_url = st.text_input(
            "URL inzerátu", placeholder="https://www.nehnutelnosti.sk/...",
            label_visibility="collapsed", key="_listing_url"
        )
    with _btn_col:
        _fetch_clicked = st.button("Načítať", use_container_width=True)

    if _fetch_clicked and _listing_url.strip():
        with st.spinner("Načítavam údaje z inzerátu..."):
            _result = fetch_listing(_listing_url.strip())
        if not _result['success']:
            st.error(_result['error'])
        else:
            _detected_cat = _result['category']
            if _detected_cat != category:
                # Category mismatch — switch segment and retry after rerun
                st.session_state.segment_toggle = 'Byty' if _detected_cat == 'byty' else 'Domy'
                st.session_state._scraper_pending = _result
                st.rerun()
            else:
                _inp, _warns, _conf = _build_scraper_input(_result['parsed'], category, mappings)
                _apply_scraper_result(_inp, _warns, _conf, _result.get('title', ''))
                st.rerun()
    elif _fetch_clicked:
        st.warning("Zadajte URL inzerátu.")

# --- Scraper info banner ---
if st.session_state.scraper_result:
    _sr = st.session_state.scraper_result
    _sr_title = _sr.get('title', 'Inzerát')
    _sr_price = _sr['input'].get('market_price', 0)
    _title_short = (_sr_title[:80] + '...') if len(_sr_title) > 80 else _sr_title
    _all_warnings = list(_sr.get('warnings', []))
    if _sr.get('confidence') == 'low':
        _all_warnings.append("Lokalita bola priradená len približne. Odporúčame overiť výber.")
    _price_html = f'<div class="scraper-price">Inzerovaná cena: {_sr_price:,} €</div>' if _sr_price > 0 else ''
    _warnings_html = ''.join(f'<div class="scraper-warning-item">⚠ {w}</div>' for w in _all_warnings)
    st.markdown(
        f'<div class="scraper-banner">'
        f'<div class="scraper-title">{_title_short}</div>'
        f'{_price_html}'
        f'{_warnings_html}'
        f'<div class="scraper-hint">Formulár bol vyplnený údajmi z inzerátu. Skontrolujte hodnoty a kliknite Odhadnúť cenu.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# --- Prefill dict (from scraper or empty) ---
_pf = {}
if st.session_state.scraper_result:
    _pf = st.session_state.scraper_result.get('input', {})

# --- Location Search (outside form — needs dynamic filtering) ---
all_locations = sorted(mappings['locations'].keys())
_sb_ver = st.session_state.get('_sb_ver', 0)
_loc_fallback = next((l for l in all_locations if l.startswith('Bratislava')), all_locations[0] if all_locations else None)
_loc_default = _pf.get('obec_cast', _loc_fallback)
if _loc_default and _loc_default not in mappings['locations']:
    _loc_default = _loc_fallback

def search_location(query: str) -> list[str]:
    if not query.strip():
        return all_locations
    q = strip_diacritics(query).lower()
    return [loc for loc in all_locations if q in strip_diacritics(loc).lower()]

location = st_searchbox(
    search_location,
    placeholder="Hľadať obec... (napr. Cadca, Zilina)",
    label="Obec / mestská časť",
    default=_loc_default,
    default_searchterm=_loc_default if _pf.get('obec_cast') else "",
    default_options=all_locations[:20],
    key=f"loc_searchbox_{category}_{_sb_ver}",
)
if location is None or location not in mappings['locations']:
    location = _loc_fallback or ""

# Determine active model region and features for form rendering
_form_region = get_region(category, location)
_form_features = set(model_sets[_form_region][0])

# Amenity toggles — only show features the active model uses
_AMENITIES = [
    ('has_lift', 'Výťah'),
    ('has_balcony', 'Balkón'),
    ('has_loggia', 'Loggia'),
    ('has_cellar', 'Pivnica'),
    ('has_garage', 'Garáž'),
    ('has_parking', 'Parkovanie'),
    ('has_terrace', 'Terasa'),
    ('has_pantry', 'Špajza'),
    ('has_warehouse', 'Sklad'),
    ('has_ac', 'Klimatizácia'),
]
_visible_amenities = [(k, l) for k, l in _AMENITIES if k in _form_features and k != 'has_lift']

# ============================================================
# SECTION 1: INPUT FORM — 2-column layout (versioned for scraper auto-fill)
# ============================================================

_form_ver = st.session_state.get('_form_ver', 0)

# --- Compute prefill defaults for selectboxes ---
_hidden_stav = {'undefined', 'Určený k demolácii', 'Developerský projekt'}
stav_options = [s for s in mappings['options']['stav_final'] if s not in _hidden_stav]

_pf_stav = _pf.get('stav_final', stav_options[0] if stav_options else 'Pôvodný stav')
if _pf_stav in _hidden_stav:
    _pf_stav = {'Developerský projekt': 'Novostavba'}.get(_pf_stav, 'Pôvodný stav')
_stav_idx = stav_options.index(_pf_stav) if _pf_stav in stav_options else 0

_hide_constr_always = {'Unknown', 'Skeleton', 'Prefab'}
_hide_constr_byty = {'Stone', 'Wood'}
_hide_constr_domy = {'Panel', 'ReinforcedConcrete'}
_hide = _hide_constr_always | (_hide_constr_byty if category == 'byty' else _hide_constr_domy)
constr_options = [c for c in mappings['options']['construction'] if c not in _hide]
constr_labels = [CONSTRUCTION_LABELS.get(c, c) for c in constr_options]

_pf_constr = _pf.get('construction', constr_options[0] if constr_options else 'Brick')
if _pf_constr in _hide:
    _pf_constr = constr_options[0] if constr_options else 'Brick'
_pf_constr_label = CONSTRUCTION_LABELS.get(_pf_constr, _pf_constr)
_constr_idx = constr_labels.index(_pf_constr_label) if _pf_constr_label in constr_labels else 0

heat_options = [h for h in mappings['options']['heating'] if h != 'unknown']
heat_labels = [HEATING_LABELS.get(h, h) for h in heat_options]

_pf_heat = _pf.get('heating', heat_options[0] if heat_options else 'central')
if _pf_heat not in heat_options:
    _pf_heat = heat_options[0] if heat_options else 'central'
_pf_heat_label = HEATING_LABELS.get(_pf_heat, _pf_heat)
_heat_idx = heat_labels.index(_pf_heat_label) if _pf_heat_label in heat_labels else 0

_vlastnictvo_opts = ['Osobné', 'Firemné']
_pf_vlast = _pf.get('vlastnictvo', 'Osobné')
_vlast_idx = _vlastnictvo_opts.index(_pf_vlast) if _pf_vlast in _vlastnictvo_opts else 0

with st.form(f"input_form_{_form_ver}"):
    form_left, form_right = st.columns(2)

    # --- Card 1: Nehnuteľnosť ---
    with form_left:
        with st.container(border=True):
            st.markdown('<p class="card-label">Nehnuteľnosť</p>', unsafe_allow_html=True)
            _fs_max = 500 if category == 'byty' else 2000
            floor_size = st.number_input(
                "Plocha (m²)", min_value=10, max_value=_fs_max,
                value=max(10, min(_fs_max, int(_pf.get('floor_size', 60)))),
                help="Úžitková plocha nehnuteľnosti."
            )
            room_count = st.number_input(
                "Počet izieb", min_value=1, max_value=10,
                value=max(1, min(10, int(_pf.get('room_count', 2)))),
            )
            stav = st.selectbox("Stav", stav_options, index=_stav_idx)
            _constr_label = st.selectbox("Konštrukcia", constr_labels, index=_constr_idx)
            construction = constr_options[constr_labels.index(_constr_label)]
            if category == 'byty':
                current_floor = st.number_input("Poschodie", min_value=0, max_value=30,
                    value=max(0, min(30, int(_pf.get('current_floor', 2)))))
                total_floors = st.number_input("Počet poschodí v budove", min_value=1, max_value=30,
                    value=max(1, min(30, int(_pf.get('total_floors', 5)))))
                has_lift = st.toggle("Výťah", value=bool(_pf.get('has_lift', False)))
                land_area = 0
                built_up_area = 0
            else:
                current_floor = 0
                total_floors = st.number_input("Počet podlaží", min_value=1, max_value=5,
                    value=max(1, min(5, int(_pf.get('total_floors', 1)))))
                land_area = st.number_input("Plocha pozemku (m²)", min_value=0, max_value=50000,
                    value=max(0, min(50000, int(_pf.get('land_area', 400)))),
                    help="Celková plocha pozemku.")
                built_up_area = st.number_input("Zastavaná plocha (m²)", min_value=0, max_value=5000,
                    value=max(0, min(5000, int(_pf.get('built_up_area', 100)))),
                    help="Plocha, ktorú zaberá samotná stavba na pozemku.")
                has_lift = False

    # --- Right Column: Ostatné + Vybavenie + Porovnanie ---
    with form_right:
        with st.container(border=True):
            st.markdown('<p class="card-label">Ostatné</p>', unsafe_allow_html=True)
            vlastnictvo = st.selectbox("Vlastníctvo", _vlastnictvo_opts, index=_vlast_idx)
            _heat_label = st.selectbox("Kúrenie", heat_labels, index=_heat_idx)
            heating = heat_options[heat_labels.index(_heat_label)]

        with st.container(border=True):
            st.markdown('<p class="card-label">Vybavenie</p>', unsafe_allow_html=True)
            _amenity_values = {}
            if _visible_amenities:
                _mid = (len(_visible_amenities) + 1) // 2
                vybavenie_cols = st.columns(2)
                with vybavenie_cols[0]:
                    for _akey, _alabel in _visible_amenities[:_mid]:
                        _amenity_values[_akey] = st.toggle(_alabel, value=bool(_pf.get(_akey, False)), key=f"am_{_akey}")
                with vybavenie_cols[1]:
                    for _akey, _alabel in _visible_amenities[_mid:]:
                        _amenity_values[_akey] = st.toggle(_alabel, value=bool(_pf.get(_akey, False)), key=f"am_{_akey}")
            has_balcony = _amenity_values.get('has_balcony', False)
            has_loggia = _amenity_values.get('has_loggia', False)
            has_cellar = _amenity_values.get('has_cellar', False)
            has_garage = _amenity_values.get('has_garage', False)
            has_parking = _amenity_values.get('has_parking', False)
            has_terrace = _amenity_values.get('has_terrace', False)

        with st.container(border=True):
            st.markdown('<p class="card-label">Porovnanie s inzerátom</p>', unsafe_allow_html=True)
            market_price = st.number_input(
                "Inzerovaná cena (€)", min_value=0,
                value=max(0, int(_pf.get('market_price', 0))),
                help="Voliteľné — zadajte cenu z inzerátu pre porovnanie s odhadom."
            )

    submitted = st.form_submit_button("Odhadnúť cenu", type="primary")

_valid = True
if submitted:
    # Edge case validations
    if category == 'byty' and current_floor > total_floors:
        st.warning("Poschodie nemôže byť vyššie ako počet poschodí v budove.")
        _valid = False
    if category == 'domy' and land_area == 0:
        st.warning("Plocha pozemku nie je zadaná. Odhad môže byť nepresný.")
    if category == 'domy' and built_up_area > land_area > 0:
        st.warning("Zastavaná plocha nemôže byť väčšia ako plocha pozemku.")
        _valid = False

if submitted and _valid:
    st.session_state.prediction_done = True
    st.session_state.scraper_result = None  # Clear prefill after form submission
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
        'has_pantry': int(_amenity_values.get('has_pantry', False)),
        'has_warehouse': int(_amenity_values.get('has_warehouse', False)),
        'has_ac': int(_amenity_values.get('has_ac', False)),
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

    # --- Prediction (select regional model set) ---
    region = get_region(category, input_data.get('obec_cast', ''))
    features, models = model_sets[region]
    X_input = process_input(input_data, mappings, features, category)
    final_price, (pred_xgb, pred_lgb, pred_cb, pred_nn) = predict_ensemble(X_input, models)
    ci_lower, ci_upper = calculate_confidence_interval(final_price, category)
    model_prices = [np.exp(pred_xgb)[0], np.exp(pred_lgb)[0], np.exp(pred_cb)[0], np.exp(pred_nn)[0]]

    # SHAP (computed once, reused across sections — always on)
    shap_values = None
    region_tag = f'byty_{region}' if category == 'byty' else 'domy'
    with st.spinner("Počítam vysvetlenie modelu..."):
        explainer = get_shap_explainer(models['xgb'], f'xgb_{region_tag}')
        shap_values = explainer(X_input)

    # ========================================================
    # SECTION 2: HERO PRICE
    # ========================================================

    st.markdown(f'<div class="hero-price">{final_price:,.0f} €</div>', unsafe_allow_html=True)

    # Verdict badge (only if market price provided)
    if market_price > 0:
        diff = market_price - final_price
        diff_pct = (diff / final_price) * 100
        if abs(diff_pct) < 5:
            badge_class, badge_text = "verdict-fair", f"Férová cena ({diff_pct:+.1f}%)"
        elif diff_pct > 20:
            badge_class, badge_text = "verdict-over", f"Výrazne nadhodnotená (+{diff_pct:.1f}%)"
        elif diff_pct > 5:
            badge_class, badge_text = "verdict-slight-over", f"Mierne nadhodnotená (+{diff_pct:.1f}%)"
        elif diff_pct < -20:
            badge_class, badge_text = "verdict-under", f"Výrazne podhodnotená ({diff_pct:.1f}%)"
        else:
            badge_class, badge_text = "verdict-good", f"Výhodná kúpa ({diff_pct:.1f}%)"
        st.markdown(
            f'<div style="text-align:center"><span class="verdict-badge {badge_class}">{badge_text}</span></div>',
            unsafe_allow_html=True
        )

    # Price context subtitle
    price_per_m2 = final_price / input_data['floor_size']
    loc_stats_hero = mappings['locations'].get(input_data['obec_cast'], {'location_score_m2': 0})
    loc_avg_m2 = loc_stats_hero['location_score_m2']
    st.markdown(
        f'<p class="hero-subtitle">{price_per_m2:,.0f} €/m² · priemer lokality {loc_avg_m2:,.0f} €/m²</p>',
        unsafe_allow_html=True
    )

    # --- Three stat cards: Interval | €/m² | Zhoda ---
    s1, s2, s3 = st.columns(3)
    _stat_desc_style = 'font-size:0.75rem;color:var(--text-secondary);margin-top:2px'
    with s1:
        mape_pct = (DASHBOARD['MAPE_BYTY'] if category == 'byty' else DASHBOARD['MAPE_DOMY']) * 100
        st.markdown(f'<div class="stat-card">'
            f'<div class="stat-label">Cenové rozpätie (±{mape_pct:.0f}%)</div>'
            f'<div class="stat-value">{ci_lower:,.0f} € — {ci_upper:,.0f} €</div>'
            f'<div style="{_stat_desc_style}">Na základe presnosti modelu</div>'
            f'</div>', unsafe_allow_html=True)
    with s2:
        _m2_diff = price_per_m2 - loc_avg_m2
        _m2_note = f'{_m2_diff:+,.0f} €/m² oproti priemeru' if loc_avg_m2 > 0 else '&nbsp;'
        st.markdown(f'<div class="stat-card">'
            f'<div class="stat-label">Cena za m²</div>'
            f'<div class="stat-value">{price_per_m2:,.0f} €/m²</div>'
            f'<div style="{_stat_desc_style}">{_m2_note}</div>'
            f'</div>', unsafe_allow_html=True)
    with s3:
        cv = np.std(model_prices) / np.mean(model_prices) if np.mean(model_prices) > 0 else 0
        if cv < 0.05:
            dot_html, agreement_label = '<span class="dot-green">●</span>', "Vysoká"
            agreement_desc = "Modely sa zhodujú"
        elif cv < 0.15:
            dot_html, agreement_label = '<span class="dot-yellow">●</span>', "Stredná"
            agreement_desc = "Mierne rozdiely medzi modelmi"
        else:
            dot_html, agreement_label = '<span class="dot-red">●</span>', "Nízka"
            agreement_desc = "Veľké rozdiely — orientujte sa podľa cenového rozpätia"
        st.markdown(f'<div class="stat-card">'
            f'<div class="stat-label">Spoľahlivosť odhadu</div>'
            f'<div class="stat-value">{dot_html} {agreement_label}</div>'
            f'<div style="{_stat_desc_style}">{agreement_desc}</div>'
            f'</div>', unsafe_allow_html=True)

    # ========================================================
    # SECTION 3: MAP + SHAP
    # ========================================================

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    map_col, shap_col = st.columns([1, 1])

    with map_col:
        with st.container(border=True):
            st.markdown('<p class="card-label">Poloha nehnuteľnosti</p>', unsafe_allow_html=True)
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
                tooltip={"text": "{name}\nCena: {price} €\nPlocha: {area} m²"}
            ), use_container_width=True)

    with shap_col:
        with st.container(border=True):
            st.markdown('<p class="card-label">Čo najviac ovplyvňuje cenu?</p>', unsafe_allow_html=True)
            if shap_values is not None:
                top_factors = get_plain_language_shap(shap_values[0], final_price, X_input=X_input, top_n=5)
                for label, eur, fmt_val in top_factors:
                    direction = "zvyšuje" if eur > 0 else "znižuje"
                    _arrow = '<span style="color:#34c759;font-size:0.9rem">▲</span>' if eur > 0 else '<span style="color:#ff3b30;font-size:0.9rem">▼</span>'
                    val_str = f" ({fmt_val})" if fmt_val else ""
                    st.markdown(f"{_arrow} **{label}**{val_str} {direction} cenu o **{abs(eur):,.0f} €**", unsafe_allow_html=True)
                st.caption("Na základe XGBoost SHAP analýzy. Hodnoty sú aproximácie vplyvu na finálnu cenu.")

    # ========================================================
    # SECTION 4: WHAT-IF ANALYSIS
    # ========================================================

    with st.container(border=True):
        st.markdown('<p class="card-label">Scenáre</p>', unsafe_allow_html=True)

        # 4a: Renovation simulation
        st.markdown("#### Simulácia rekonštrukcie")
        current_stav = input_data['stav_final']
        if current_stav not in ('Kompletná rekonštrukcia', 'Novostavba', 'Developerský projekt'):
            input_renovated = input_data.copy()
            input_renovated['stav_final'] = 'Kompletná rekonštrukcia'
            X_renovated = process_input(input_renovated, mappings, features, category)
            price_renovated, _ = predict_ensemble(X_renovated, models)
            increase = price_renovated - final_price
            st.metric("Potenciálna cena po rekonštrukcii", f"{price_renovated:,.0f} €", delta=f"+{increase:,.0f} €")
            st.caption("Odhad nezahŕňa náklady na rekonštrukciu.")
        else:
            st.info("Nehnuteľnosť je už v top stave (Novostavba / Rekonštrukcia).")

        # 4b: Location comparison
        st.markdown("---")
        st.markdown("#### Porovnanie lokalít")

        all_locations = sorted(mappings['locations'].keys())
        whatif_search = st.text_input("Hľadať lokality na porovnanie...", value="", key="whatif_search",
                                       placeholder="napr. Cadca, Zilina, Presov",
                                       help="Funguje aj bez diakritiky. Vybrané lokality zostanú zachované.")
        whatif_all = [loc for loc in all_locations if loc != input_data['obec_cast']]
        # Keep already-selected items in options so they don't disappear on search change
        already_selected = st.session_state.get("whatif_locations", [])
        if whatif_search.strip():
            whatif_query = strip_diacritics(whatif_search).lower()
            whatif_filtered = [loc for loc in whatif_all if whatif_query in strip_diacritics(loc).lower()]
        else:
            whatif_filtered = whatif_all
        # Prepend selected items that aren't in filtered results
        for loc in already_selected:
            if loc in whatif_all and loc not in whatif_filtered:
                whatif_filtered.insert(0, loc)
        target_locs = st.multiselect(
            "Porovnať cenu v iných lokalitách (max 3):",
            whatif_filtered,
            max_selections=3,
            key="whatif_locations"
        )

        if target_locs:
            comparison_results = []
            for loc in target_locs:
                input_loc = input_data.copy()
                input_loc['obec_cast'] = loc
                # Use the correct regional model for this location
                loc_region = get_region(category, loc)
                loc_features, loc_models = model_sets[loc_region]
                X_loc = process_input(input_loc, mappings, loc_features, category)
                price_loc, _ = predict_ensemble(X_loc, loc_models)
                loc_stats_comp = mappings['locations'].get(loc, {'location_score_m2': 0, 'lat': 48.15, 'lon': 17.11})
                shap_diff_top = []
                _cross_region = loc_region != region
                loc_tag = f'byty_{loc_region}' if category == 'byty' else 'domy'
                # SHAP diff only works when both locations use the same model (same region)
                if shap_values is not None and not _cross_region:
                    shap_loc = get_shap_explainer(loc_models['xgb'], f'xgb_{loc_tag}')(X_loc)
                    shap_diff_top = get_shap_diff_top(shap_values[0], shap_loc[0], price_loc)
                comparison_results.append({
                    'location': loc, 'price': price_loc,
                    'price_m2': price_loc / input_data['floor_size'],
                    'loc_avg_m2': loc_stats_comp['location_score_m2'],
                    'lat': loc_stats_comp['lat'], 'lon': loc_stats_comp['lon'],
                    'shap_diff': shap_diff_top,
                    'cross_region': _cross_region,
                })

            n_cols = 1 + len(comparison_results)
            cols = st.columns(n_cols)
            with cols[0]:
                st.markdown(f"**{input_data['obec_cast']}** (originál)")
                st.metric("Cena", f"{final_price:,.0f} €")
                st.caption(f"{price_per_m2:,.0f} €/m²")

            for i, comp in enumerate(comparison_results):
                with cols[i + 1]:
                    delta = comp['price'] - final_price
                    st.markdown(f"**{comp['location']}**")
                    st.metric("Cena", f"{comp['price']:,.0f} €", delta=f"{delta:+,.0f} €")
                    st.caption(f"{comp['price_m2']:,.0f} €/m²")
                    if comp['shap_diff']:
                        for lbl, eur_d in comp['shap_diff']:
                            _arrow = '<span style="color:#34c759">▲</span>' if eur_d > 0 else '<span style="color:#ff3b30">▼</span>'
                            st.markdown(f"{_arrow} {lbl}: **{eur_d:+,.0f} €**", unsafe_allow_html=True)
                    elif comp.get('cross_region'):
                        st.caption("Iný región — porovnanie faktorov nie je dostupné")

            # Multi-pin map
            loc_orig = mappings['locations'].get(input_data['obec_cast'], {'lat': 48.15, 'lon': 17.11})
            pin_colors = [[200, 30, 0, 180], [30, 130, 200, 180], [30, 180, 30, 180], [255, 150, 0, 180]]
            all_pins = [{
                'lat': loc_orig['lat'], 'lon': loc_orig['lon'],
                'name': input_data['obec_cast'] + ' (originál)',
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
                tooltip={"text": "{name}\nCena: {price} €"}
            ), use_container_width=True)

    # ========================================================
    # SECTION 5: DETAILED ANALYSIS (collapsed)
    # ========================================================

    with st.expander("Podrobná analýza"):
        st.markdown("#### SHAP diagram")
        st.caption("Diagram ukazuje, ako jednotlivé faktory posúvajú odhad ceny od priemeru. "
                   "Červené stĺpce zvyšujú cenu, modré ju znižujú.")
        shap_model_name = st.selectbox(
            "Model pre vysvetlenie:",
            ["XGBoost", "LightGBM", "CatBoost"],
            key="shap_model_detail",
        )
        model_key_map = {"XGBoost": 'xgb', "LightGBM": 'lgb', "CatBoost": 'cb'}
        detail_key = model_key_map[shap_model_name]
        detail_explainer = get_shap_explainer(models[detail_key], f'{detail_key}_{region_tag}')
        detail_shap = detail_explainer(X_input)
        # Translate feature names to Slovak for display
        detail_shap_row = detail_shap[0]
        detail_shap_row.feature_names = [translate_feature_name(f) for f in detail_shap_row.feature_names]
        try:
            shap.plots.waterfall(detail_shap_row, show=False, max_display=10)
        except IndexError:
            pass  # SHAP tick-label coloring bug — plot is already rendered
        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        st.pyplot(fig, use_container_width=False)
        plt.clf()

        st.markdown("#### Porovnanie modelov")
        details = pd.DataFrame({
            'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'Neural Network', 'Ensemble (Final)'],
            'Predikcia (€)': model_prices + [final_price],
            'Odchýlka': [f"{((p - final_price) / final_price) * 100:+.1f}%" for p in model_prices] + ['—']
        })
        st.table(details.style.format({'Predikcia (€)': '{:,.0f}'}))

        median_price = np.median(model_prices)
        clipped = [min(max(p, median_price * 0.3), median_price * 3.0) for p in model_prices]
        chart_df = pd.DataFrame({
            'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'Neural Network', 'Ensemble'],
            'Predikcia (€)': clipped + [final_price]
        })
        st.bar_chart(chart_df.set_index('Model'))

    # ========================================================
    # SECTION 6: TECHNICAL DETAILS (collapsed)
    # ========================================================

    with st.expander("Technické detaily modelu"):
        mape_display = DASHBOARD['MAPE_BYTY'] if category == 'byty' else DASHBOARD['MAPE_DOMY']
        if category == 'byty':
            st.markdown(f"""
**Architektúra:** Stacking Ensemble (XGBoost + LightGBM + CatBoost + NN → Ridge meta-learner)

**Regionálny split:** Bratislava a zvyšok Slovenska majú oddelené modely (rôzna cenová dynamika)

**Dataset:** 12 815 bytov (3 467 BA + 9 348 mimo BA), po regionálnej filtrácii anomálií

**Validačné metriky (10-fold CV):**
| Región | MAPE | MAE | R² | Atribútov |
|---|---|---|---|---|
| Bratislava | 6.4% | 17 658 € | 0.950 | 36 |
| Mimo BA | 9.8% | 16 084 € | 0.887 | 46 |

**Feature Selection:** XGBoost importance (threshold 0.005) — 69 → 36/46 atribútov

**Tréning:** Marec 2026 · Dáta: Február 2026
""")
        else:
            st.markdown(f"""
**Architektúra:** Stacking Ensemble (XGBoost + LightGBM + CatBoost + NN → Ridge meta-learner)

**Dataset:** 12 426 domov (po regionálnej filtrácii anomálií)

**Validačné metriky (10-fold CV):**
| Metrika | Hodnota |
|---|---|
| MAPE | {mape_display*100:.1f}% |
| MAE | 51 559 € |
| R² | 0.759 |

**Atribútov:** 37 (z 68 po feature selection)

**Feature Selection:** XGBoost importance (threshold 0.005) — 68 → 37 atribútov

**Tréning:** Marec 2026 · Dáta: Február 2026
""")

    # Store results in session state
    st.session_state.prediction_results = {
        'final_price': final_price, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'X_input': X_input, 'pred_xgb': pred_xgb, 'pred_lgb': pred_lgb,
        'pred_cb': pred_cb, 'pred_nn': pred_nn, 'model_prices': model_prices,
    }

# ========================================================
# FEEDBACK + FOOTER
# ========================================================
with st.expander("Spätná väzba"):
    feedback_type = st.selectbox("Typ:", ["Návrh na zlepšenie", "Chyba / bug", "Nepresný odhad", "Iné"], key="fb_type")
    feedback_text = st.text_area("Vaša správa:", height=120, key="fb_text",
                                  placeholder="Opíšte váš návrh alebo problém...")
    if st.button("Odoslať spätnú väzbu", type="primary", key="fb_submit"):
        if feedback_text.strip():
            import requests as _req
            _formspree_url = "https://formspree.io/f/mjgaaejd" 
            try:
                _resp = _req.post(_formspree_url, json={
                    "email": "anonymous@feedback.form",
                    "message": f"[{feedback_type}] {feedback_text}",
                }, headers={"Accept": "application/json"})
                if _resp.ok:
                    st.success("Spätná väzba bola odoslaná. Ďakujeme!")
                else:
                    st.error("Nepodarilo sa odoslať. Skúste neskôr alebo nás kontaktujte na ajesensky8@gmail.com.")
            except Exception:
                st.error("Nepodarilo sa odoslať. Skúste neskôr alebo nás kontaktujte na ajesensky8@gmail.com.")
        else:
            st.warning("Prosím, napíšte správu.")

st.markdown("""
<div class="app-footer">
    <strong>Proof of Concept</strong> · Diplomová práca, UNIZA 2026<br>
    Odhad vychádza z inzerovaných cien a slúži ako orientácia, nie ako znalecký posudok.<br>
    Model nezohľadňuje dispozíciu, výhľad, hlučnosť okolia ani aktuálny stav interiéru.<br><br>
    <a href="https://github.com/adamJesensky/slovak-real-estate-predictor/issues">Nahlásiť chybu</a>
</div>
""", unsafe_allow_html=True)
