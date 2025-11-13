import streamlit as st
import pandas as pd
import os
import numpy as np

DATA_PATH = "data"

def load_data(file_name):
    try:
        file_path = os.path.join(DATA_PATH, file_name)
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"No se pudo cargar el archivo {file_name}. Error: {e}")
        return pd.DataFrame()

tabs = st.tabs([
    "Datos crudos",                
    "Transformaciones",            
    "Tratamiento de outliers",     
    "Modelo de demanda/generación",
    "Calificación de modelos",     
    "Comparación y conclusiones",  
    "Fuentes"                      
])

# 1. Datos crudos
with tabs[0]:
    st.header("Datos crudos")
    df_crisis = load_data("fechas_riesgo_crisis_energetica.csv")
    df_outliers = load_data("outliers_demanda_altos.csv")
    st.subheader("Fechas riesgo crisis energética")
    if not df_crisis.empty:
        st.dataframe(df_crisis)
    else:
        st.warning("No hay datos de crisis energética.")
    st.subheader("Outliers demanda altos")
    if not df_outliers.empty:
        st.dataframe(df_outliers)
    else:
        st.warning("No hay datos de outliers.")

# 2. Transformaciones
with tabs[1]:
    st.header("Transformaciones de datos")
    st.markdown("""
    - Conversión de la columna de fechas al formato `datetime`.
    - Imputación de valores nulos utilizando interpolación lineal.
    - Ejemplo de transformación:
    """)
    if not df_crisis.empty and 'fecha' in df_crisis.columns:
        df_temp = df_crisis.copy()
        df_temp['fecha'] = pd.to_datetime(df_temp['fecha'], errors='coerce')
        st.dataframe(df_temp.head())
    else:
        st.warning("No se puede mostrar transformación: falta la columna 'fecha' en crisis energética.")

# 3. Tratamiento de outliers
with tabs[2]:
    st.header("Tratamiento de outliers")
    st.markdown("""
    - Identificación de outliers mediante el método de puntaje z.
    """)
    if not df_outliers.empty:
        # Verificamos si la columna 'valor' existe
        valor_col = None
        for col in df_outliers.columns:
            col_lower = col.lower()
            if col_lower in ['valor', 'demanda', 'value']:
                valor_col = col
                break
        if valor_col:
            mean = df_outliers[valor_col].mean()
            std = df_outliers[valor_col].std()
            df_outliers['zscore'] = (df_outliers[valor_col] - mean) / std
            df_outliers['outlier'] = np.abs(df_outliers['zscore']) > 3
            n_outliers = df_outliers['outlier'].sum()
            st.success(f"Se identificaron {n_outliers} outliers con z > 3.")
            st.dataframe(df_outliers[df_outliers['outlier']])
        else:
            st.error("No se encontró una columna numérica válida ('valor' o 'demanda') en outliers_demanda_altos.csv.")
    else:
        st.warning("No hay datos para análisis de outliers.")

# 4. Modelo demanda/generación
with tabs[3]:
    st.header("Modelo de demanda o generación")
    st.markdown("""
    - Aquí puedes cargar tu modelo en formato `.joblib` y realizar predicciones sobre los datos cargados.
    """)
    uploaded_model = st.file_uploader("Carga tu modelo .joblib", type=["joblib"])
    if uploaded_model and valor_col:
        try:
            import joblib
            model = joblib.load(uploaded_model)
            y_pred = model.predict(df_outliers[[valor_col]])
            st.write("Predicciones del modelo (primeros 10 valores):")
            st.write(y_pred[:10])
        except Exception as e:
            st.error(f"Error al ejecutar el modelo: {e}")

# 5. Calificación de modelos
with tabs[4]:
    st.header("Calificación de los modelos")
    st.markdown("""
    - MAE, RMSE, MAPE, sMAPE, R², etc.  
    - Sube los resultados/calculados o inclúyelos aquí en forma de tabla/markdown según lo analizado.
    """)

# 6. Comparación entre generación y demanda
with tabs[5]:
    st.header("Comparación generación vs. demanda hasta 2030")
    st.markdown("""
    - Agrega una tabla o gráfica de comparación y síntesis de conclusiones.
    """)

# 7. Fuentes
with tabs[6]:
    st.header("Fuentes utilizadas")
    st.markdown("""
    - UPME, Atlas Renewable Energy, SITTCA, DNP, Caracol Radio, El Colombiano, SER Colombia, SEI, Climatetracker Latam, Invest in Colombia, entre otras oficiales y especializadas.
    """)

st.sidebar.title("Navegación")
st.sidebar.info("Proyecto Ciencia de Datos Energía Colombia")

