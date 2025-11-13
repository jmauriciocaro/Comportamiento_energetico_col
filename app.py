import streamlit as st
import pandas as pd
import os

DATA_PATH = "data"

def load_data(filename):
    try:
        file_path = os.path.join(DATA_PATH, filename)
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"No se pudo cargar {filename}: {e}")
        return pd.DataFrame()

# Opciones de la navbar moderna
sections = [
    "Datos crudos",
    "Transformaciones",
    "Tratamiento de outliers",
    "Modelo de demanda/generación",
    "Calificación de modelos",
    "Comparación y conclusiones",
    "Fuentes"
]

# Navbar en sidebar
st.sidebar.title("Menú de navegación")
selected_section = st.sidebar.radio("Ir a sección:", sections)

# 1. DATOS CRUDOS
if selected_section == "Datos crudos":
    st.markdown("<h1 style='margin-bottom: 2rem;'>Datos crudos</h1>", unsafe_allow_html=True)

    # Demanda
    st.markdown("<h2>Demanda eléctrica (muestra)</h2>", unsafe_allow_html=True)
    df_demanda = load_data("Demanda.csv")
    if not df_demanda.empty:
        st.dataframe(df_demanda.head(10))
        st.download_button(
            label="Descargar Demanda.csv",
            data=df_demanda.to_csv(index=False).encode('utf-8'),
            file_name="Demanda.csv",
            mime="text/csv"
        )
    else:
        st.warning("No se pudo cargar Demanda.csv.")

    # Generación
    st.markdown("<h2>Generación eléctrica (muestra)</h2>", unsafe_allow_html=True)
    df_generacion = load_data("Generacion.csv")
    if not df_generacion.empty:
        st.dataframe(df_generacion.head(10))
        st.download_button(
            label="Descargar Generacion.csv",
            data=df_generacion.to_csv(index=False).encode('utf-8'),
            file_name="Generacion.csv",
            mime="text/csv"
        )
    else:
        st.warning("No se pudo cargar Generacion.csv.")

# 2. TRANSFORMACIONES
elif selected_section == "Transformaciones":
    st.header("Transformaciones")
    st.info("Aquí se mostrarán los pasos y ejemplos de transformación de los datos.")

# 3. TRATAMIENTO DE OUTLIERS
elif selected_section == "Tratamiento de outliers":
    st.header("Tratamiento de outliers")
    st.info("Aquí se mostrarán los métodos y resultados para tratar outliers.")

# 4. MODELO DEMANDA/GENERACIÓN
elif selected_section == "Modelo de demanda/generación":
    st.header("Modelo de demanda/generación")
    st.info("Aquí puedes cargar y probar tus modelos.")

# 5. CALIFICACIÓN DE MODELOS
elif selected_section == "Calificación de modelos":
    st.header("Calificación de los modelos")
    st.info("Aquí se mostrarán las métricas de calidad de los modelos.")

# 6. COMPARACIÓN Y CONCLUSIONES
elif selected_section == "Comparación y conclusiones":
    st.header("Comparación generación vs demanda y conclusiones")
    st.info("Aquí se mostrarán comparaciones gráficas y texto conclusivo.")

# 7. FUENTES
elif selected_section == "Fuentes":
    st.header("Fuentes utilizadas")
    st.markdown("""
    UPME, Atlas Renewable Energy, SITTCA, DNP, Caracol Radio, El Colombiano, SER Colombia, SEI, Climatetracker Latam, Invest in Colombia y otras fuentes especializadas.
    """)

st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por [Tu Nombre]")


