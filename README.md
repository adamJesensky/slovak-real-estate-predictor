# Odhad Ceny Nehnuteľností — Slovensko

Predikcia cien bytov a domov na Slovensku pomocou strojového učenia.

**[Vyskúšajte živú aplikáciu →](https://cena-nehnutelnosti.streamlit.app)**

## Čo to robí?

Zadajte parametre nehnuteľnosti (lokalita, plocha, stav, ...) a nástroj odhadne jej trhovú cenu
na základe 10 000+ reálnych inzerátov z nehnutelnosti.sk.

## Ako to funguje?

- **4 modely:** XGBoost, LightGBM, CatBoost, neurónová sieť
- **Stacking Ensemble:** Ridge meta-learner kombinuje predikcie
- **SHAP vysvetlenia:** Vidíte, čo cenu ovplyvňuje najviac
- **What-If analýza:** Porovnajte cenu v rôznych lokalitách

## Presnosť

| Kategória | MAPE | MAE | R² |
|-----------|------|-----|-----|
| Byty | 13.1% | 27,388 € | 0.856 |
| Domy | 26.9% | 55,639 € | 0.739 |

## Tech Stack

Python, Streamlit, XGBoost, LightGBM, CatBoost, PyTorch, SHAP, scikit-learn

## Diplomová práca

Žilinská univerzita v Žiline, 2026.

---

*Odhad slúži ako orientácia, nie ako znalecký posudok.*
