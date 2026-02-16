
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib 
from xgboost import XGBClassifier

# --- 1. CONFIGURACIÓN DE RUTAS DINÁMICAS ---
# Esto asegura que encuentre los archivos en la carpeta /src sin importar desde dónde lances el comando
recurso_path = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    csv_path = os.path.join(recurso_path, 'df_final.csv')
    df = pd.read_csv(csv_path)
    # Buscamos la columna de fecha para ordenar los registros
    col_fecha = 'Date' if 'Date' in df.columns else ('MatchDate' if 'MatchDate' in df.columns else None)
    if col_fecha:
        df[col_fecha] = pd.to_datetime(df[col_fecha])
    return df, col_fecha

@st.cache_resource
def load_model():
    model_path = os.path.join(recurso_path, 'modelo_futbol_xgboost.pkl')
    return joblib.load(model_path)

# Función de ayuda para buscar columnas ignorando mayúsculas/minúsculas
def get_stat(df_row, key):
    options = [key, key.lower(), key.capitalize(), key.replace('Elo', 'ELO'), key.upper()]
    for opt in options:
        if opt in df_row.index:
            return df_row[opt]
    return 0

# --- 2. CARGA INICIAL DE DATOS Y MODELO ---
try:
    df_matches, nombre_col_fecha = load_data()
    modelo_final = load_model()
except Exception as e:
    st.error(f"❌ Error al cargar archivos: {e}")
    st.stop()

# --- 3. CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Predictor Pro", page_icon="⚽", layout="wide")
st.markdown("""
    <style>
    .stProgress > div > div > div > div { background-color: #2ecc71; }
    .main { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR (Interfaz de Usuario) ---
with st.sidebar:
    st.title("🛡️ Panel de Control")
    st.subheader("Selección de Partido")
    
    # Determinamos si usamos nombres o códigos para mostrar en el selector
    col_h = 'HomeTeam' if 'HomeTeam' in df_matches.columns else 'HomeTeam_Code'
    col_a = 'AwayTeam' if 'AwayTeam' in df_matches.columns else 'AwayTeam_Code'
    
    lista_equipos = sorted(df_matches[col_h].unique())
    
    eq_l = st.selectbox("Equipo Local", lista_equipos, index=0)
    eq_v = st.selectbox("Equipo Visitante", lista_equipos, index=1)

    # Extraemos el último registro histórico disponible para cada equipo
    if nombre_col_fecha:
        stats_h = df_matches[df_matches[col_h] == eq_l].sort_values(nombre_col_fecha).iloc[-1]
        stats_v = df_matches[df_matches[col_a] == eq_v].sort_values(nombre_col_fecha).iloc[-1]
    else:
        stats_h = df_matches[df_matches[col_h] == eq_l].iloc[-1]
        stats_v = df_matches[df_matches[col_a] == eq_v].iloc[-1]

    st.success("✅ Datos sincronizados")

# --- 5. PREPARACIÓN DE DATOS (Alineado con tu entrenamiento) ---
# Aquí incluimos todas las columnas que el modelo "aprendió", incluyendo las 14 que faltaban
stats_dict = {
    'Form3Home': get_stat(stats_h, 'Form3Home'),
    'Form5Home': get_stat(stats_h, 'Form5Home'),
    'Form3Away': get_stat(stats_v, 'Form3Away'),
    'Form5Away': get_stat(stats_v, 'Form5Away'),
    'OddHome': get_stat(stats_h, 'OddHome'),
    'OddDraw': get_stat(stats_h, 'OddDraw'),
    'OddAway': get_stat(stats_h, 'OddAway'),
    
    # Columnas técnicas detectadas en tu error anterior
    'MaxHome': get_stat(stats_h, 'MaxHome'),
    'MaxDraw': get_stat(stats_h, 'MaxDraw'),
    'MaxAway': get_stat(stats_h, 'MaxAway'),
    'Over25': get_stat(stats_h, 'Over25'),
    'Under25': get_stat(stats_h, 'Under25'),
    'MaxOver25': get_stat(stats_h, 'MaxOver25'),
    'MaxUnder25': get_stat(stats_h, 'MaxUnder25'),
    'HandiSize': get_stat(stats_h, 'HandiSize'),
    'HandiHome': get_stat(stats_h, 'HandiHome'),
    'HandiAway': get_stat(stats_h, 'HandiAway'),
    'MatchTime_is_missing': get_stat(stats_h, 'MatchTime_is_missing'),
    'Division_Code': get_stat(stats_h, 'Division_Code'),
    'HomeTeam_Code': get_stat(stats_h, 'HomeTeam_Code'),
    'AwayTeam_Code': get_stat(stats_v, 'AwayTeam_Code'),
    
    # Variables de ELO e ingeniería
    'EloTotal': get_stat(stats_h, 'EloTotal'),
    'EloAdvantage': get_stat(stats_h, 'EloAdvantage'),
    
    # Variables temporales
    'Anio': 2026,
    'Mes': 2,
    'Dia_Semana': 5,
    'Hour_sin': 0.5,
    'Hour_cos': 0.8
}

df_input = pd.DataFrame([stats_dict])

# --- 6. EJECUCIÓN DE PREDICCIÓN CON FILTRO DE COLUMNAS ---
try:
    # Obtenemos exactamente qué columnas quiere el modelo y en qué orden
    cols_modelo = modelo_final.get_booster().feature_names
    df_input_final = df_input[cols_modelo]
    
    # Probabilidades reales del XGBoost
    probs = modelo_final.predict_proba(df_input_final)[0]
    
    # --- 7. DISEÑO DEL CUERPO PRINCIPAL ---
    st.title("⚽ Dashboard de Predicción de Resultados")
    st.caption(f"Encuentro analizado: **{eq_l} vs {eq_v}**")
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        with st.container(border=True):
            st.subheader("📋 Datos del Encuentro")
            # Mostramos solo las variables clave para no saturar
            resumen = {
                "Cuota Local": stats_dict['OddHome'],
                "Forma Local (5)": stats_dict['Form5Home'],
                "Forma Visitante (5)": stats_dict['Form5Away']
            }
            st.json(resumen)
            st.write("Variables procesadas:", len(cols_modelo))

    with col2:
        with st.container(border=True):
            st.subheader("🔮 Probabilidades Calculadas")
            clases = [f'Victoria {eq_l}', 'Empate', f'Victoria {eq_v}']
            iconos = ['🏠', '🤝', '🚀']
            
            for i in range(len(clases)):
                c_txt, c_prb = st.columns([2, 1])
                c_txt.write(f"### {iconos[i]} {clases[i]}")
                c_prb.write(f"## {probs[i]*100:.1f}%")
                st.progress(float(probs[i]))
            
            st.divider()
            ganador_idx = np.argmax(probs)
            st.success(f"### 🎯 Resultado Sugerido: **{clases[ganador_idx]}**")

except Exception as e:
    st.error(f"❌ Error en la predicción: {e}")
    st.info("Esto sucede si el modelo espera columnas que no están en el diccionario stats_dict.")