import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="IA Pesquera Cantábrico", layout="centered")
st.title("IA Pesquera del Cantábrico 🌊🐙")

# Cargar dataset real (ya generado)
df = pd.read_csv("dataset_pesca_cantabrico.csv")

# Variables predictoras
features = ["temp_agua","oleaje","viento","lluvia","presion","coef_marea"]

# Etiqueta: especie con mayor captura simulada
df["especie"] = df[["captura_pulpo","captura_lubina","captura_percebe"]].idxmax(axis=1)

X = df[features]
y = df["especie"]

# Entrenar modelo de IA
modelo = RandomForestClassifier(n_estimators=80, random_state=42)
modelo.fit(X, y)

st.markdown("### Ajusta las condiciones ambientales para predecir la mejor especie a capturar")

# Interfaz de usuario
temp = st.slider("Temperatura agua (ºC)", 10.0, 24.0, 18.0)
ola = st.slider("Oleaje (m)", 0.0, 4.0, 1.5)
viento = st.slider("Viento (km/h)", 0.0, 40.0, 10.0)
lluvia = st.slider("Lluvia (mm)", 0.0, 20.0, 1.0)
presion = st.slider("Presión atmosférica (hPa)", 990.0, 1040.0, 1015.0)
marea = st.slider("Coeficiente de marea", 50, 120, 90)

if st.button("Predecir pesca"):
    pred = modelo.predict([[temp, ola, viento, lluvia, presion, marea]])
    st.success(f"Mejor opción hoy: {pred[0]}")

st.markdown("---")
st.markdown("### Datos de ejemplo de tu dataset")
st.dataframe(df.head(10))
