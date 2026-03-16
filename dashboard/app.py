
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
    'location_score_m2': 'Lokalita (cena/m²)',
    'floor_size': 'Plocha',
    'log_area': 'Plocha',
    'condition_score': 'Stav nehnuteľnosti',
    'dist_bratislava': 'Vzdialenosť od Bratislavy',
    'dist_nearest_city': 'Vzdialenosť od krajského mesta',
    'room_count': 'Počet izieb',
    'has_lift': 'Výťah',
    'relative_floor': 'Relatívne poschodie',
    'utilities_score': 'Inžinierske siete',
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
}

def shap_to_eur(shap_value, final_price):
    """Convert SHAP value (ln-price space) to EUR impact."""
    return final_price * (1 - np.exp(-shap_value))

def get_plain_language_shap(shap_values_row, final_price, top_n=5):
    """Return list of (feature_label, eur_impact) sorted by |impact|."""
    values = shap_values_row.values
    feature_names = shap_values_row.feature_names
    impacts = []
    for sv, fname in zip(values, feature_names):
        eur = shap_to_eur(sv, final_price)
        label = FEATURE_LABELS.get(fname, fname)
        impacts.append((label, eur, fname))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    seen_labels = set()
    unique = []
    for label, eur, fname in impacts:
        if label not in seen_labels:
            seen_labels.add(label)
            unique.append((label, eur))
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
        'Čiastočná rekonštrukcia': 2, 'Pôvodný stav': 1, 'Vo výstavbe': 4, 'Iný stav': 0,
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
# MAIN APP — Single Scrollable Flow (Apple HIG-inspired)
# ============================================================

st.set_page_config(
    page_title="Odhad ceny nehnuteľnosti | Slovensko",
    layout="wide",
    page_icon="🏡",
    menu_items={
        'About': 'Odhad ceny bytov a domov na Slovensku pomocou strojového učenia. Diplomová práca, UNIZA 2026.'
    }
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .section-divider { border-top: 1px solid #e0e0e0; margin: 2rem 0; }
    .dot-green { color: #34c759; font-size: 1.3rem; }
    .dot-yellow { color: #ff9500; font-size: 1.3rem; }
    .dot-red { color: #ff3b30; font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)

st.title("🏡 Predikcia Ceny Nehnuteľností")

# --- Session State ---
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'input_data' not in st.session_state:
    st.session_state.input_data = None

# --- Sidebar ---
with st.sidebar:
    category = st.selectbox(
        "Segment", ["byty", "domy"],
        help="Vyberte typ nehnuteľnosti: byty (flats) alebo domy (houses)."
    )
    st.divider()
    show_shap = st.checkbox("Zobraziť SHAP analýzu", value=True)
    show_whatif = st.checkbox("Povoliť What-If simulácie", value=True)

with st.spinner("Načítavam modely... Prosím čakajte."):
    mappings, features, models = load_resources(category)


# ============================================================
# SECTION 1: INPUT FORM
# ============================================================

st.subheader("Parametre nehnuteľnosti")

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Nehnuteľnosť**")
        floor_size = st.number_input(
            "Plocha (m²)", min_value=10, max_value=500, value=60,
            help="Úžitková plocha nehnuteľnosti. Jeden z najdôležitejších faktorov ceny."
        )
        room_count = st.number_input(
            "Počet izieb", min_value=1, max_value=10, value=2,
            help="Počet obytných miestností."
        )
        stav = st.selectbox(
            "Stav", mappings['options']['stav_final'],
            help="Stav nehnuteľnosti. Novostavba vs pôvodný stav môže znamenať rozdiel 15-30%."
        )
        construction = st.selectbox(
            "Konštrukcia", mappings['options']['construction'],
            help="Typ konštrukcie budovy."
        )
        if category == 'byty':
            current_floor = st.number_input(
                "Poschodie", min_value=0, max_value=30, value=2,
                help="Aktuálne poschodie (0 = prízemie). Vyššie poschodia bez výťahu znižujú cenu."
            )
            total_floors = st.number_input(
                "Počet poschodí v budove", min_value=1, max_value=30, value=5,
                help="Celkový počet poschodí v bytovom dome."
            )
            has_lift = st.checkbox("Výťah", help="Prítomnosť výťahu v budove.")
            land_area = 0
            built_up_area = 0
        else:
            current_floor = 0
            total_floors = st.number_input(
                "Počet podlaží", min_value=1, max_value=5, value=1,
                help="Počet podlaží domu."
            )
            land_area = st.number_input(
                "Plocha pozemku (m²)", min_value=0, max_value=5000, value=400,
                help="Celková plocha pozemku."
            )
            built_up_area = st.number_input(
                "Zastavaná plocha (m²)", min_value=0, max_value=500, value=100,
                help="Zastavaná plocha domu."
            )
            has_lift = False

    with col2:
        st.markdown("**Lokalita**")
        location = st.selectbox(
            "Lokalita (Obec/Časť)", sorted(mappings['locations'].keys()),
            help="Obec alebo mestská časť. Lokalita je dominantný faktor — rovnaký byt v BA stojí 3× viac než v regióne."
        )
        vlastnictvo = st.selectbox(
            "Vlastníctvo",
            ['Osobné', 'Družstevné', 'Firemné', 'Obecné', 'Štátne', 'Iné', 'Nevysporiadané', 'Unknown'],
            help="Typ vlastníctva (osobné, družstevné, atď.)."
        )
        heating = st.selectbox(
            "Kúrenie", mappings['options']['heating'],
            help="Typ vykurovania nehnuteľnosti."
        )

    with col3:
        st.markdown("**Vybavenie**")
        has_balcony = st.checkbox("Balkón", help="Prítomnosť balkónu.")
        has_loggia = st.checkbox("Loggia", help="Prítomnosť loggie.")
        has_cellar = st.checkbox("Pivnica", help="Prítomnosť pivnice/skladu.")
        has_garage = st.checkbox("Garáž", help="Prítomnosť garáže.")
        has_parking = st.checkbox("Parkovanie", help="Parkovacie miesto.")
        has_terrace = st.checkbox("Terasa", help="Prítomnosť terasy.")
        st.markdown("**Inžinierske siete**")
        has_gas = st.checkbox("Plyn", value=True, help="Prípojka plynu.")
        has_water = st.checkbox("Voda", value=True, help="Prípojka vody.")
        has_electricity = st.checkbox("Elektrina", value=True, help="Prípojka elektriny.")
        has_sewerage = st.checkbox("Kanalizácia", value=True, help="Prípojka kanalizácie.")

    st.markdown("---")
    market_price = st.number_input(
        "Inzerovaná cena (€)", min_value=0, value=0,
        help="Voliteľné — zadajte cenu z inzerátu pre porovnanie s odhadom modelu."
    )

    submitted = st.form_submit_button("Predikovať Cenu", type="primary")

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
        'has_gas': int(has_gas),
        'has_water': int(has_water),
        'has_electricity': int(has_electricity),
        'has_sewerage': int(has_sewerage),
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

    # SHAP (computed once, reused across sections)
    shap_values = None
    if show_shap:
        with st.spinner("Počítam vysvetlenie modelu..."):
            explainer = get_shap_explainer(models['xgb'], f'xgb_{category}')
            shap_values = explainer(X_input)

    # ========================================================
    # SECTION 2: RESULTS HERO
    # ========================================================

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Price metrics
    res1, res2, res3 = st.columns(3)
    with res1:
        st.metric("Predikovaná cena", f"{final_price:,.0f} €")
    with res2:
        st.metric("Interval neistoty", f"{ci_lower:,.0f} € — {ci_upper:,.0f} €")
    with res3:
        mape_val = DASHBOARD['MAPE_BYTY'] if category == 'byty' else DASHBOARD['MAPE_DOMY']
        st.metric("MAPE segmentu", f"{mape_val*100:.1f}%")

    # Price per m² context
    price_per_m2 = final_price / input_data['floor_size']
    loc_stats_hero = mappings['locations'].get(input_data['obec_cast'], {'location_score_m2': 0})
    loc_avg_m2 = loc_stats_hero['location_score_m2']
    st.caption(f"Predikovaná cena za m²: **{price_per_m2:,.0f} €/m²** (priemer lokality: {loc_avg_m2:,.0f} €/m²)")

    # Model agreement indicator
    cv = np.std(model_prices) / np.mean(model_prices) if np.mean(model_prices) > 0 else 0
    if cv < 0.05:
        agreement_html = '<span class="dot-green">●</span>'
        agreement_label = "Vysoká zhoda modelov"
    elif cv < 0.15:
        agreement_html = '<span class="dot-yellow">●</span>'
        agreement_label = "Mierna zhoda modelov"
    else:
        agreement_html = '<span class="dot-red">●</span>'
        agreement_label = "Nízka zhoda — vyššia neistota"

    st.markdown(f'{agreement_html} **{agreement_label}** (CV: {cv:.1%})', unsafe_allow_html=True)
    st.caption("Signál konzistencie predikcie — miera zhody medzi 4 nezávislými modelmi.")

    # Price verdict badge
    if market_price > 0:
        diff = market_price - final_price
        diff_pct = (diff / final_price) * 100

        if abs(diff_pct) < 5:
            st.info(f"✅ **Férová cena** — rozdiel {diff_pct:+.1f}% ({diff:+,.0f} €)")
        elif diff_pct > 20:
            st.error(f"⚠️ **Výrazne nadhodnotená** — o {diff_pct:.1f}% ({diff:,.0f} €) vyššia než odhad")
        elif diff_pct > 5:
            st.warning(f"↗️ **Mierne nadhodnotená** — o {diff_pct:.1f}% ({diff:,.0f} €) vyššia než odhad")
        elif diff_pct < -20:
            st.warning(f"⚠️ **Výrazne podhodnotená** — o {abs(diff_pct):.1f}% ({abs(diff):,.0f} €) nižšia — pozor na skryté vady!")
        else:
            st.success(f"↘️ **Výhodná kúpa** — o {abs(diff_pct):.1f}% ({abs(diff):,.0f} €) nižšia než odhad")

    # ========================================================
    # SECTION 3: MAP + PLAIN-LANGUAGE SHAP
    # ========================================================

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    map_col, shap_col = st.columns([1, 1])

    with map_col:
        st.markdown("### 📍 Poloha nehnuteľnosti")
        loc_name = input_data['obec_cast']
        loc_viz = mappings['locations'].get(loc_name, {'lat': 48.1486, 'lon': 17.1077})

        map_style_choice = st.selectbox("Štýl mapy", ["Dark", "Light"], key="map_style_main")

        map_data = pd.DataFrame({
            'lat': [loc_viz['lat']], 'lon': [loc_viz['lon']],
            'name': [loc_name], 'price': [f"{final_price:,.0f}"],
            'area': [input_data['floor_size']]
        })
        st.pydeck_chart(pdk.Deck(
            map_style=MAP_STYLES[map_style_choice],
            initial_view_state=pdk.ViewState(
                latitude=loc_viz['lat'], longitude=loc_viz['lon'],
                zoom=13, pitch=45),
            layers=[pdk.Layer("ScatterplotLayer", data=map_data,
                get_position='[lon, lat]', get_color='[200, 30, 0, 160]',
                get_radius=200, pickable=True)],
            tooltip={"text": "{name}\nCena: {price} €\nPlocha: {area} m²"}
        ), use_container_width=True)

    with shap_col:
        st.markdown("### 💡 Čo najviac ovplyvňuje cenu?")
        if show_shap and shap_values is not None:
            top_factors = get_plain_language_shap(shap_values[0], final_price, top_n=5)

            for label, eur in top_factors:
                direction = "zvyšuje" if eur > 0 else "znižuje"
                icon = "🔺" if eur > 0 else "🔻"
                st.markdown(f"{icon} **{label}** {direction} cenu o **{abs(eur):,.0f} €**")

            st.caption("Na základe XGBoost SHAP analýzy. Hodnoty sú aproximácie vplyvu na finálnu cenu.")
        else:
            st.info("SHAP analýza je vypnutá. Povoľte ju v bočnom paneli.")

    # ========================================================
    # SECTION 4: WHAT-IF ANALYSIS
    # ========================================================

    if show_whatif:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔄 What-If Analýza")

        # 4a: Reconstruction simulation
        st.markdown("#### 🛠️ Simulácia rekonštrukcie")
        current_stav = input_data['stav_final']
        if current_stav not in ('Kompletná rekonštrukcia', 'Novostavba', 'Developerský projekt'):
            input_renovated = input_data.copy()
            input_renovated['stav_final'] = 'Kompletná rekonštrukcia'
            X_renovated = process_input(input_renovated, mappings, features, category)
            price_renovated, _ = predict_ensemble(X_renovated, models)
            increase = price_renovated - final_price
            st.metric("Potenciálna cena po rekonštrukcii", f"{price_renovated:,.0f} €", delta=f"+{increase:,.0f} €")
        else:
            st.info("Nehnuteľnosť je už v top stave (Novostavba / Rekonštrukcia).")

        # 4b: Location comparison
        st.markdown("---")
        st.markdown("#### 🌍 Porovnanie lokalít")
        st.write("Porovnajte cenu rovnakej nehnuteľnosti v rôznych lokalitách (max. 3).")

        all_locations = sorted(mappings['locations'].keys())
        target_locs = st.multiselect(
            "Vyberte lokality na porovnanie:",
            [loc for loc in all_locations if loc != input_data['obec_cast']],
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

                # SHAP diff
                shap_diff_top = []
                if show_shap and shap_values is not None:
                    shap_loc = get_shap_explainer(models['xgb'], f'xgb_{category}')(X_loc)
                    shap_diff_top = get_shap_diff_top(shap_values[0], shap_loc[0], price_loc)

                comparison_results.append({
                    'location': loc,
                    'price': price_loc,
                    'price_m2': price_loc / input_data['floor_size'],
                    'loc_avg_m2': loc_stats_comp['location_score_m2'],
                    'lat': loc_stats_comp['lat'],
                    'lon': loc_stats_comp['lon'],
                    'shap_diff': shap_diff_top,
                })

            # Side-by-side columns
            n_cols = 1 + len(comparison_results)
            cols = st.columns(n_cols)

            with cols[0]:
                st.markdown(f"**📍 {input_data['obec_cast']}** (originál)")
                st.metric("Cena", f"{final_price:,.0f} €")
                st.caption(f"{price_per_m2:,.0f} €/m² (lok. priemer: {loc_avg_m2:,.0f} €/m²)")

            for i, comp in enumerate(comparison_results):
                with cols[i + 1]:
                    delta = comp['price'] - final_price
                    st.markdown(f"**📍 {comp['location']}**")
                    st.metric("Cena", f"{comp['price']:,.0f} €", delta=f"{delta:+,.0f} €")
                    st.caption(f"{comp['price_m2']:,.0f} €/m² (lok. priemer: {comp['loc_avg_m2']:,.0f} €/m²)")

                    if comp['shap_diff']:
                        st.markdown("**Zmena faktorov:**")
                        for lbl, eur_d in comp['shap_diff']:
                            icon = "🔺" if eur_d > 0 else "🔻"
                            st.markdown(f"{icon} {lbl}: **{eur_d:+,.0f} €**")

            # Multi-pin map
            loc_orig = mappings['locations'].get(input_data['obec_cast'], {'lat': 48.15, 'lon': 17.11})
            pin_colors = [[200, 30, 0, 180], [30, 130, 200, 180], [30, 180, 30, 180], [255, 150, 0, 180]]

            all_pins = [{
                'lat': loc_orig['lat'], 'lon': loc_orig['lon'],
                'name': input_data['obec_cast'] + ' (originál)',
                'price': f"{final_price:,.0f}",
                'color': pin_colors[0],
            }]
            for i, comp in enumerate(comparison_results):
                all_pins.append({
                    'lat': comp['lat'], 'lon': comp['lon'],
                    'name': comp['location'],
                    'price': f"{comp['price']:,.0f}",
                    'color': pin_colors[i + 1],
                })

            pins_df = pd.DataFrame(all_pins)
            avg_lat = pins_df['lat'].mean()
            avg_lon = pins_df['lon'].mean()
            lat_range = pins_df['lat'].max() - pins_df['lat'].min()
            lon_range = pins_df['lon'].max() - pins_df['lon'].min()
            max_range = max(lat_range, lon_range, 0.01)
            zoom = max(5, min(13, 8 - np.log2(max_range + 0.001)))

            whatif_map_style = st.selectbox("Štýl mapy", ["Dark", "Light"], key="map_style_whatif")

            st.pydeck_chart(pdk.Deck(
                map_style=MAP_STYLES[whatif_map_style],
                initial_view_state=pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=zoom, pitch=30),
                layers=[pdk.Layer("ScatterplotLayer", data=pins_df,
                    get_position='[lon, lat]', get_color='color',
                    get_radius=300, pickable=True)],
                tooltip={"text": "{name}\nCena: {price} €"}
            ), use_container_width=True)

    # ========================================================
    # SECTION 5: DETAILED ANALYSIS (collapsed)
    # ========================================================

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    with st.expander("📊 Podrobná analýza (pre odborníkov)"):
        if show_shap:
            st.markdown("#### SHAP Waterfall")
            shap_model_name = st.selectbox(
                "Model pre vysvetlenie:",
                ["XGBoost", "LightGBM", "CatBoost"],
                key="shap_model_detail",
                help="Neurónová sieť nie je podporovaná pre SHAP analýzu."
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

            st.caption("**Červené (+):** zvyšujú cenu oproti priemeru. **Modré (-):** znižujú cenu oproti priemeru.")

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

    with st.expander("⚙️ Technické detaily modelu"):
        mape_display = DASHBOARD['MAPE_BYTY'] if category == 'byty' else DASHBOARD['MAPE_DOMY']
        st.markdown(f"""
**Architektúra:** Stacking Ensemble (XGB + LGB + CB + NN → Ridge)

**Dataset:** {'10,773 bytov' if category == 'byty' else '11,060 domov'} (po filtrácii anomálií)

**Počet atribútov:** {'73' if category == 'byty' else '72'}

**Validačné metriky ({category}):**
| Metrika | Hodnota |
|---|---|
| MAE | {'27,388 €' if category == 'byty' else '55,639 €'} |
| MAPE | {mape_display*100:.1f}% |
| R² | {'0.856' if category == 'byty' else '0.739'} |

**Feature Selection:** RFECV {'30' if category == 'byty' else '37'}/{'73' if category == 'byty' else '72'} → <0.6% MAE nárast

**Tréning:** Február 2026
""")

    # Store results in session state
    st.session_state.prediction_results = {
        'final_price': final_price, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'X_input': X_input, 'pred_xgb': pred_xgb, 'pred_lgb': pred_lgb,
        'pred_cb': pred_cb, 'pred_nn': pred_nn, 'model_prices': model_prices,
    }

# ========================================================
# FOOTER: About / Trust Section
# ========================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem 0;">
    <strong>O projekte</strong><br>
    Model bol natrénovaný na 10 000+ reálnych inzerátoch z nehnutelnosti.sk (február 2026).<br>
    Strojové učenie: XGBoost + LightGBM + CatBoost + neurónová sieť → Stacking Ensemble.<br>
    Presnosť: MAPE 13.1% (byty), 26.9% (domy). Odhad slúži ako orientácia, nie ako znalecký posudok.<br>
    Diplomová práca — Žilinská univerzita v Žiline, 2026.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding-bottom: 2rem;">
    <a href="https://github.com/adamJesensky/slovak-real-estate-predictor/issues" target="_blank"
       style="color: #60a5fa; text-decoration: none; font-size: 0.85rem;">
        💬 Máte spätnú väzbu alebo ste našli chybu? Napíšte nám →
    </a>
</div>
""", unsafe_allow_html=True)
