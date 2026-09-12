# 🌾 Assistant Agronomique — Recommandation de culture & Prédiction de rendement

Projet de machine learning combinant deux modèles indépendants pour l'aide à la décision agricole, déployé via Streamlit.

## Fonctionnalités

- **Recommandation de culture** : à partir des paramètres du sol et du climat (azote, phosphore, potassium, température, humidité, pH, pluviométrie), le modèle recommande la culture la plus adaptée parmi 22 cultures possibles.
- **Prédiction de rendement** : à partir de la région, du type de sol, de la culture, des conditions climatiques et des intrants (engrais, irrigation), le modèle estime le rendement attendu (tonnes/hectare).

## Datasets

- [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) (Kaggle) — 2200 observations, 22 cultures
- [Agriculture Crop Yield Dataset](https://www.kaggle.com/datasets) (Kaggle) — 1 000 000 observations, 6 cultures

Les deux datasets sont traités **indépendamment** (pas de clé de fusion fiable entre eux) : deux modèles séparés cohabitent dans la même application.

## Résultats

| Modèle | Type | Métrique | Score |
|---|---|---|---|
| Recommandation de culture | Classification (Random Forest) | Accuracy | 99.55% |
| Prédiction de rendement | Régression (Random Forest) | R² / RMSE | 0.912 / 0.50 t/ha |

**Variables les plus importantes pour le rendement** : pluviométrie (63%), usage d'engrais (21%), irrigation (13%).

## Stack technique

- Python, Pandas, NumPy, Scikit-learn
- Streamlit (interface et déploiement)
- Git LFS (fichiers `.pkl` volumineux)

## Structure du projet

```
crop-recommendation-yield/
├── data/                          # CSV sources (non versionnés, voir .gitignore)
├── model/                         # Modèles entraînés (.pkl, via Git LFS)
│   ├── crop_recommendation_model.pkl
│   ├── yield_prediction_model.pkl
│   └── label_encoders.pkl
├── train_models.py                # Script d'entraînement des 2 modèles
├── app.py                         # Application Streamlit
├── requirements.txt
└── .gitignore
```

## Installation et lancement

```bash
# Cloner le repo
git clone https://github.com/hebaelLLL/crop-recommendation-yield.git
cd crop-recommendation-yield

# Installer les dépendances
pip install -r requirements.txt

# Entraîner les modèles (place d'abord tes CSV dans data/)
python train_models.py

# Lancer l'application
streamlit run app.py
```

## Auteur

Hiba El Guarouani — Étudiante ingénieure, spécialisation Data Science Appliquée à l'Agronomie, IAV Hassan II (Rabat)
