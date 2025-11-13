import streamlit as st
import pandas as pd
import os
import numpy as np

DATA_PATH = "data"
NOTEBOOK_PATH = "notebook"

@st.cache_data
def load_data(file_name):
    file_path = os.path.join(DATA_PATH, file_name)
    return pd.read_csv(file_path)

# Añade aquí el nombre de tu tercer dataset si lo tienes.
third_dataset = None
try:
    third_dataset = load_data("demanda.csv")
except Exception:
    third_dataset = None

tabs = st.tabs([
    "Datos crudos",                # 1
    "Transformaciones",            # 2
    "Tratamiento de outliers",     # 3
    "Modelo de demanda/generación",# 4
    "Calificación de modelos",     # 5
    "Comparación y conclusiones",  # 6
    "Fuentes"                      # 7
])

# 1. Datos crudos
with tabs[0]:
    st.header("Datos crudos")
    df_crisis = load_data("fechas_riesgo_crisis_energetica.csv")
    df_outliers = load_data("outliers_demanda_altos.csv")
    st.subheader("Fechas riesgo crisis energética")
    st.dataframe(df_crisis)
    st.subheader("Outliers demanda altos")
    st.dataframe(df_outliers)
    if third_dataset is not None:
        st.subheader("Demanda")
        st.dataframe(third_dataset)

# 2. Transformaciones
with tabs[1]:
    st.header("Transformaciones de datos")
    st.markdown("""
    - Se realizó conversión de la columna de fechas al formato `datetime`.
    - Los valores nulos fueron imputados usando interpolación lineal.
    - Se agregaron columnas para análisis temporal: día, mes, año.
    - Los valores de demanda fueron normalizados usando min-max scaling.
    Ejemplo de estructura transformada:
    """)
    if third_dataset is not None:
        td = third_dataset.copy()
        if "fecha" in td.columns:
            td["fecha"] = pd.to_datetime(td["fecha"])
            td["dia"] = td["fecha"].dt.day
            td["mes"] = td["fecha"].dt.month
            td["año"] = td["fecha"].dt.year
        st.dataframe(td.head())

# 3. Tratamiento de outliers
with tabs[2]:
    st.header("Tratamiento de outliers")
    st.markdown("""
    - Se identificaron valores atípicos mediante el método de desviación estándar (> 3σ).
    - Para tratamiento, se dejaron sin remover pero se etiquetaron con una nueva columna 'outlier' = True.
    - Los valores extremos pueden ser visualizados a continuación.
    """)
    df_outliers['valor_zscore'] = (df_outliers['valor'] - df_outliers['valor'].mean()) / df_outliers['valor'].std()
    df_outliers['outlier'] = np.abs(df_outliers['valor_zscore']) > 3
    st.dataframe(df_outliers[df_outliers['outlier']])

# 4. Modelo demanda/generación
with tabs[3]:
    st.header("Modelo de demanda o generación")
    st.markdown("""
    - Aquí puedes subir tu modelo entrenado en formato `.joblib` para realizar predicciones con los datos crudos o transformados.
    - Ejemplo: modelo Prophet, RandomForest, etc.
    """)
    uploaded_model = st.file_uploader("Carga tu modelo .joblib", type=["joblib"])
    if uploaded_model:
        import joblib
        model = joblib.load(uploaded_model)
        st.write("Modelo cargado correctamente. Suba datos para predicción en las secciones previas.")
        # Ejemplo básico de predicción:
        # Si tu modelo espera una columna 'valor' en third_dataset
        if third_dataset is not None:
            try:
                y_pred = model.predict(third_dataset[['valor']])
                st.write("Predicciones del modelo (primeros 5):")
                st.write(y_pred[:5])
            except Exception as e:
                st.write(f"No se pudo ejecutar predicción: {e}")

# 5. Calificación de modelos
with tabs[4]:
    st.header("Calificación de los modelos")
    st.markdown("""
    - Se evaluó el desempeño con las métricas:
        - MAE (Error absoluto medio)
        - RMSE (Raíz del error cuadrático medio)
        - MAPE (Error porcentual absoluto medio)
        - sMAPE (Error porcentual absoluto medio simétrico)
        - R² (Coeficiente de determinación)
    - Ejemplo de comparación (cargar resultados) :
    """)
    # Puedes cargar aquí los resultados finales del modelo, por ejemplo un archivo .csv con métricas.

# 6. Comparación entre generación y demanda
with tabs[5]:
    st.header("Comparación generación vs. demanda hasta 2030")
    st.markdown("""
    - Se comparan los valores proyectados de generación y demanda anual.
    - Graficar ambas series permite visualizar posibles brechas o déficits.  
    - Conclusión: Es necesario seguir aumentando la capacidad renovable y optimizando la infraestructura.
    """)
    # Ejemplo de gráfica (necesitas datos de ambos lados):
    # st.line_chart(df_comparacion) si tienes un dataframe con columnas 'año', 'demanda', 'generacion'

# 7. Fuentes
with tabs[6]:
    st.header("Fuentes utilizadas")
    st.markdown("""
    - UPME, Atlas Renewable Energy, SITTCA, DNP, Caracol Radio, El Colombiano, SER Colombia, SEI, Climatetracker Latam, Invest in Colombia, entre otras oficiales y especializadas.
    - Consulta el informe notebook para detalles y citas.
    """)

st.sidebar.title("Navegación")
st.sidebar.info("Proyecto Ciencia de Datos Energía Colombia")
