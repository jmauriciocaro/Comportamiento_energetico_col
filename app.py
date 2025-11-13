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

    st.markdown("""
    En el notebook de modelos, las transformaciones principales son:
    - Conversión de la columna de fecha al formato AAAA-MM-DD.
    - Conversión del valor de energía a GWh si estaba en otra unidad (por ejemplo, MWh → GWh).
    - La tabla final deseada es: Fecha | Valor (en GWh).
    
    Ejemplo de la estructura final de la tabla:
    """)

    # Ejemplo sintético
    import pandas as pd
    ejemplo = pd.DataFrame({
        "Fecha": pd.date_range("2025-01-01", periods=5, freq="D"),
        "Valor_MWh": [89000, 102500, 95000, 87000, 120000],  # valores originales en MWh
    })
    ejemplo["Fecha"] = ejemplo["Fecha"].dt.strftime("%Y-%m-%d")  # Formato AAAA-MM-DD
    ejemplo["Valor"] = (ejemplo["Valor_MWh"] / 1000).round(2)    # Convertido a GWh, redondeado a 2 decimales
    st.dataframe(ejemplo[["Fecha", "Valor"]].rename(columns={"Valor": "Valor (GWh)"}))

    # Ejemplo sintético de tabla transformada
    import pandas as pd
    ejemplo = pd.DataFrame({
        "Fecha": pd.date_range("2025-01-01", periods=5, freq="D"),
        "Valor": [89000, 102500, 95000, 87000, 120000],    # valores en MWh
    })
    ejemplo["Valor_GWh"] = ejemplo["Valor"] / 1000

    st.dataframe(ejemplo)

# 3. TRATAMIENTO DE OUTLIERS
elif selected_section == "Tratamiento de outliers":
    st.markdown("<h1>Tratamiento de outliers</h1>", unsafe_allow_html=True)

    st.markdown("<h2>Investigaciones</h2>", unsafe_allow_html=True)
    st.markdown("""
    En la revisión de los outliers no se evidenciaron eventos extraordinarios relacionados con factores externos. 
    El criterio de tratamiento aplicado fue:  
    Si la demanda diaria superaba en más de 1.3 veces el valor medio histórico, se reemplazó ese dato por el promedio de la serie.  
    Esto se justifica por posibles acumulaciones de demanda en el día, o errores humanos en la digitación de datos.
    """)

    # Cargar archivo de outliers
    df_outliers = load_data("outliers_demanda_altos.csv")
    # Detectar columna numérica para graficar
    valor_col = None
    for col in df_outliers.columns:
        if df_outliers[col].dtype in ['float64', 'int64']:
            valor_col = col
            break

    from numpy import nan
    if not df_outliers.empty and valor_col:
        mean = df_outliers[valor_col].mean()
        threshold = mean * 1.3
        # Nueva columna para marcar si es outlier
        df_outliers["outlier"] = df_outliers[valor_col] > threshold
        # Reemplazo de outliers por la media
        df_outliers["valor_tratado"] = df_outliers[valor_col].where(~df_outliers["outlier"], mean)

        st.markdown("<h3>Tabla con tratamiento de outliers</h3>", unsafe_allow_html=True)
        st.dataframe(df_outliers[[df_outliers.columns[0], valor_col, "valor_tratado", "outlier"]].head(10))

        st.markdown("<h3>Gráfica: demanda original vs tratada</h3>", unsafe_allow_html=True)
        grafico = df_outliers[[df_outliers.columns[0], valor_col, "valor_tratado"]].set_index(df_outliers.columns[0]).head(50)
        st.line_chart(grafico)
    else:
        st.warning("No se encontraron valores numéricos en el archivo de outliers.")


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


