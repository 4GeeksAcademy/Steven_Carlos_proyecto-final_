
import streamlit as st
import pandas as pd
import numpy as np
import joblib 
from xgboost import XGBClassifier

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Predicador de Fútbol Pro", 
    page_icon="⚽",
    layout="wide"
)

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
    <style>
    .stProgress > div > div > div > div { background-color: #2ecc71; }
    .main { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (MENÚ LATERAL) ---
with st.sidebar:
    st.title("Menu")
    st.subheader("Paises, dentro ligas")
    st.subheader("Temporada")
    st.subheader("equipos en racha")
    st.subheader("(%) de riesgo, apuesta segura ")
    st.subheader("filtro de cuotas")
    st.subheader("limites over/under.Filtros específicos para mercados de Over2.5 y Under2.5. ")
    
    st.markdown("---")
    
    # Usamos columnas dentro del sidebar para que se vea ordenado
    st.write("**Equipo Local**")
    home_elo = st.slider("Home ELO", 1000, 3000, 1500, help="Nivel de fuerza del equipo local")
    odd_home = st.number_input("Cuota Local", 1.0, 20.0, 2.10)
    form_home = st.slider("Forma Local (Últimos 5)", 0.0, 1.0, 0.5)

    st.markdown("---")
    
    st.write("**Equipo Visitante**")
    away_elo = st.slider("Away ELO", 1000, 3000, 1500, help="Nivel de fuerza del equipo visitante")
    odd_away = st.number_input("Cuota Visitante", 1.0, 20.0, 3.20)
    form_away = st.slider("Forma Visitante (Últimos 5)", 0.0, 1.0, 0.5)

    st.markdown("---")
    odd_draw = st.number_input("Cuota Empate", 1.0, 10.0, 3.10)

# --- LÓGICA DE DATOS ---
data = {
    'HomeElo': home_elo, 'AwayElo': away_elo,
    'OddHome': odd_home, 'OddDraw': odd_draw, 'OddAway': odd_away,
    'Form5Home': form_home, 'Form5Away': form_away,
    'Anio': 2025, 'Mes': 2, 'Dia_Semana': 5,
    'Hora_Seno': 0.5, 'Hora_Coseno': 0.8
}
df_input = pd.DataFrame(data, index=[0])

# --- CUERPO PRINCIPAL ---
st.title("⚽ Dashboard de Predicción de Resultados")
st.info("Ajusta los valores en el menú lateral para ver cómo cambian las probabilidades en tiempo real.")

col_main1, col_main2 = st.columns([1, 2], gap="large")

with col_main1:
    with st.container(border=True):
        st.subheader("📋 Resumen de Entrada")
        # Mostramos los datos de forma más estética
        st.dataframe(df_input.T.rename(columns={0: 'Valor'}), use_container_width=True)

with col_main2:
    with st.container(border=True):
        st.subheader("🔮 Predicción del Algoritmo")
        
        # Simulación de probabilidades (Aquí irá tu modelo real después)
        probs = [0.45, 0.25, 0.30] 
        clases = ['Victoria Local (H)', 'Empate (D)', 'Victoria Visitante (A)']
        iconos = ['🏠', '🤝', '🚀']
        
        # Mostrar resultados con un diseño más limpio
        for i in range(len(clases)):
            col_text, col_prob = st.columns([2, 1])
            col_text.write(f"### {iconos[i]} {clases[i]}")
            col_prob.write(f"## {probs[i]*100:.1f}%")
            st.progress(probs[i])
        
        st.divider()
        ganador = clases[np.argmax(probs)]
        st.success(f"### 🎯 Resultado sugerido: **{ganador}**")

# --- FOOTER / ANÁLISIS ---
st.markdown("---")
col_f1, col_f2 = st.columns(2)

with col_f1:
    st.subheader("📈 Análisis de Importancia")
    st.write("El modelo detecta que el **Diferencial de ELO** es el factor clave en este encuentro.")
    # Gráfico de barras simple nativo de streamlit
    chart_data = pd.DataFrame(
        [home_elo, away_elo],
        index=["Local", "Visitante"],
        columns=["Puntos ELO"]
    )
    st.bar_chart(chart_data)

with col_f2:
    st.subheader("📌 Nota Informativa")
    st.caption("""
    Este dashboard es una herramienta de análisis estadístico. 
    Las probabilidades se basan en el rendimiento histórico y algoritmos de Machine Learning.
    Recuerda que en el fútbol nada es 100% seguro.
    """)