# Real Estate AI Valuator 🏠

Projekt pre automatizovaný odhad cien nehnuteľností na Slovensku pomocou metód strojového učenia a hlbokého učenia. Súčasť diplomovej práce (2026).

## 📌 O Projekte

Cieľom projektu je vytvoriť robustný systém, ktorý dokáže predikovať trhovú cenu bytov a domov na základe ich parametrov (lokalita, plocha, stav, vybavenie). Systém využíva pokročilé ML techniky (XGBoost, LightGBM, CatBoost) a Deep Learning (TabNet), pričom dosahuje presnosť MAE ~26,000 € pre byty v rámci celého Slovenska.

### Kľúčové Vlastnosti
- **Automatizovaný zber dát:** Scraper s podporou Tor siete a rotáciou identít.
- **Pokročilý Feature Engineering:** Extrakcia GPS, text mining z popisov, target encoding lokalít.
- **Ensemble Learning:** Stacking model kombinujúci 4 rôzne algoritmy.
- **Interaktívny Dashboard:** Webová aplikácia (Streamlit) pre oceňovanie v reálnom čase s mapovými vizualizáciami.
- **Vysvetliteľnosť (XAI):** Interpretácia predikcií pomocou SHAP hodnôt (prečo model určil danú cenu).

## 🛠️ Technológie

- **Jazyk:** Python 3.13
- **Data:** Pandas, NumPy, SQLAlchemy, PostgreSQL
- **ML/DL:** XGBoost, LightGBM, CatBoost, PyTorch (TabNet), Scikit-learn
- **Vizualizácia:** Matplotlib, Seaborn, PyDeck, SHAP
- **Web:** Streamlit
- **Scraping:** Requests, BeautifulSoup, PySocks (Tor)

## 📂 Štruktúra Projektu

```
.
├── analysis/           # Skripty pre EDA, feature engineering a trénovanie modelov
├── dashboard/          # Streamlit aplikácia
├── database/           # Definícia DB schémy a migrácie
├── datasets/           # Spracované CSV datasety (v4.0)
├── docs/               # Dokumentácia k diplomovej práci (Markdown)
├── models/             # Uložené natrénované modely (.joblib, .pth)
├── scraper/            # Logika zberu dát (Pipeline, Parsers, TorClient)
├── config.py           # Centrálna konfigurácia
└── requirements.txt    # Závislosti
```

## 🚀 Inštalácia a Spustenie

### 1. Prerekvizity
- Python 3.10+
- PostgreSQL databáza
- Tor (voliteľné, pre scraping)

### 2. Inštalácia závislostí
```bash
pip install -r requirements.txt
```

### 3. Konfigurácia
Vytvorte súbor `.env` (alebo upravte `config.py`) s pripojením k databáze:
```
DATABASE_URL=postgresql://user:password@localhost:5432/real_estate_db
```

### 4. Spustenie Dashboardu
```bash
streamlit run dashboard/app.py
```

### 5. Reprodukcia Experimentov
Pre spustenie trénovacej pipeline:
```bash
python analysis/model_training_v3.py
```

## 📊 Výsledky Modelov (Byty)

| Model | MAE (€) | R² |
|-------|---------|----|
| **Stacking Ensemble** | **26,623** | **0.86** |
| XGBoost | 27,009 | 0.86 |
| LightGBM | 27,243 | 0.86 |
| TabNet (DL) | 44,676 | 0.66 |
| Linear Regression | 63,000 | 0.45 |

## 📜 Licencia
Tento projekt slúži výhradne na akademické a výskumné účely. Dataset nie je určený na komerčné použitie.
