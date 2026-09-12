"""
app.py
======
Application Streamlit à 2 onglets :

Onglet 1 - "Recommander une culture"
    L'utilisateur entre les paramètres du sol/climat (N, P, K, température,
    humidité, pH, pluviométrie) -> le modèle 1 recommande la culture la
    plus adaptée.

Onglet 2 - "Prédire le rendement"
    L'utilisateur choisit une culture + conditions (région, sol, climat,
    engrais, irrigation) -> le modèle 2 estime le rendement (t/ha).

Usage :
    streamlit run app.py

Prérequis : avoir lancé train_models.py avant, pour générer les fichiers
dans model/ (crop_recommendation_model.pkl, yield_prediction_model.pkl,
label_encoders.pkl).
"""

import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Assistant Agronomique", page_icon="🌾", layout="centered")


# ---------------------------------------------------------------------------
# Chargement des modèles (mis en cache pour ne pas recharger à chaque clic)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_crop_recommendation_model():
    with open("model/crop_recommendation_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_yield_model_and_encoders():
    with open("model/yield_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


crop_model = load_crop_recommendation_model()
yield_model, encoders = load_yield_model_and_encoders()


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.title("🌾 Assistant Agronomique")

tab1, tab2 = st.tabs(["Recommander une culture", "Prédire le rendement"])


# ------------------- ONGLET 1 : recommandation de culture -------------------
with tab1:
    st.subheader("Quelle culture planter selon mon sol et mon climat ?")

    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Azote (N)", min_value=0, max_value=150, value=50)
        P = st.number_input("Phosphore (P)", min_value=0, max_value=150, value=50)
        K = st.number_input("Potassium (K)", min_value=0, max_value=210, value=50)
        temperature = st.number_input("Température (°C)", min_value=0.0, max_value=50.0, value=25.0)
    with col2:
        humidity = st.number_input("Humidité (%)", min_value=0.0, max_value=100.0, value=60.0)
        ph = st.number_input("pH du sol", min_value=0.0, max_value=14.0, value=6.5)
        rainfall = st.number_input("Pluviométrie (mm)", min_value=0.0, max_value=300.0, value=100.0)

    if st.button("Recommander une culture", key="btn_recommend"):
        input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                 columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        prediction = crop_model.predict(input_df)[0]
        st.success(f"Culture recommandée : **{prediction}**")

        # Bonus : probabilités sur les top 3 cultures
        proba = crop_model.predict_proba(input_df)[0]
        top3_idx = proba.argsort()[-3:][::-1]
        st.write("Top 3 des cultures les plus adaptées :")
        for idx in top3_idx:
            st.write(f"- {crop_model.classes_[idx]} ({proba[idx]*100:.1f}%)")


# ------------------- ONGLET 2 : prédiction de rendement -------------------
with tab2:
    st.subheader("Quel rendement attendre pour ma culture ?")

    region = st.selectbox("Région", encoders["Region"].classes_)
    soil_type = st.selectbox("Type de sol", encoders["Soil_Type"].classes_)
    crop = st.selectbox("Culture", encoders["Crop"].classes_)
    weather = st.selectbox("Condition météo", encoders["Weather_Condition"].classes_)

    col1, col2 = st.columns(2)
    with col1:
        rainfall_mm = st.number_input("Pluviométrie (mm)", min_value=0.0, max_value=1500.0, value=800.0)
        temp_c = st.number_input("Température moyenne (°C)", min_value=0.0, max_value=45.0, value=25.0)
        days_to_harvest = st.number_input("Jours avant récolte", min_value=1, max_value=365, value=110)
    with col2:
        fertilizer_used = st.checkbox("Engrais utilisé", value=True)
        irrigation_used = st.checkbox("Irrigation utilisée", value=True)

    if st.button("Prédire le rendement", key="btn_yield"):
        # Encodage des variables catégorielles avec les MÊMES encodeurs
        # que ceux utilisés lors de l'entraînement (essentiel pour la cohérence)
        region_enc = encoders["Region"].transform([region])[0]
        soil_enc = encoders["Soil_Type"].transform([soil_type])[0]
        crop_enc = encoders["Crop"].transform([crop])[0]
        weather_enc = encoders["Weather_Condition"].transform([weather])[0]

        input_df = pd.DataFrame([[
            region_enc, soil_enc, crop_enc, rainfall_mm, temp_c,
            int(fertilizer_used), int(irrigation_used), weather_enc, days_to_harvest
        ]], columns=[
            "Region", "Soil_Type", "Crop", "Rainfall_mm", "Temperature_Celsius",
            "Fertilizer_Used", "Irrigation_Used", "Weather_Condition", "Days_to_Harvest"
        ])

        prediction = yield_model.predict(input_df)[0]
        st.success(f"Rendement estimé : **{prediction:.2f} tonnes/hectare**")
