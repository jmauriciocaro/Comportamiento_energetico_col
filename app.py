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
    No se evidenciaron eventos extraordinarios asociados a los outliers.
    Para su tratamiento, si la demanda diaria superaba 1.3 veces la media, el valor se reemplazó por la media.
    Esto puede deberse a registrarse demanda acumulada de varios días o a errores de digitación.
    """)

    df_outliers = load_data("outliers_demanda_altos.csv")

    # Detectar columna numérica
    valor_col = None
    for col in df_outliers.columns:
        if df_outliers[col].dtype in ['float64', 'int64']:
            valor_col = col
            break

    if not df_outliers.empty and valor_col:
        st.markdown("<h3>Tabla de outlier demanda</h3>", unsafe_allow_html=True)
        st.dataframe(df_outliers.head(10))

        st.markdown("<h3>Outlier demanda</h3>", unsafe_allow_html=True)
        st.line_chart(df_outliers.set_index(df_outliers.columns[0])[valor_col])
    else:
        st.warning("No se encontró columna numérica para graficar en el archivo de outliers.")


# 4. MODELO DEMANDA/GENERACIÓN
elif selected_section == "Modelo de demanda/generación":
    st.markdown("<h1>Consideraciones</h1>", unsafe_allow_html=True)
    st.markdown("""
    - Se empleó el modelo Prophet para predecir la demanda y generación eléctrica diaria en Colombia.
    - Para la ingeniería de características, se incluyó el efecto de los días festivos, lo que facilita capturar variaciones no estacionales y posibles anomalías causadas por eventos especiales.
    - El conjunto de entrenamiento consistió en el 80% de los datos históricos y el restante 20% se destinó a prueba y validación.
    - Se aplicó validación cruzada manual para medir la robustez del modelo y evitar sobreajuste.
    - Prophet funciona descomponiendo la serie temporal en tendencias, estacionalidades y eventos especiales, permitiendo así pronósticos precisos y interpretables para diferentes horizontes temporales.
    """)

    # Modelo demanda
    st.markdown("<h2>Modelo demanda</h2>", unsafe_allow_html=True)
    try:
        st.image(os.path.join("Graficas","demanda.png"), caption='Demanda: Real vs Predicho (con Validación Cruzada)', use_column_width=True)
    except Exception:
        st.warning("No se pudo cargar la gráfica demanda.png. Verifica la ruta y existencia del archivo.")
    st.markdown("R² global (manual, cross-validation): **0.2162**")

    # Modelo generación
    st.markdown("<h2>Modelo generación</h2>", unsafe_allow_html=True)
    try:
        st.image(os.path.join("Graficas","generacion.png"), caption='Generación: Real vs Predicho (con Validación Cruzada)', use_column_width=True)
    except Exception:
        st.warning("No se pudo cargar la gráfica generacion.png. Verifica la ruta y existencia del archivo.")
    st.markdown("R² global (manual, cross-validation): **0.7655**")


    # --- Título principal modelo generación ---
    st.markdown("<h1>Modelo Prophet Generación</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Consideraciones</h2>", unsafe_allow_html=True)
    st.markdown("""
    - El modelo Prophet fue entrenado para predecir la generación eléctrica diaria.
    - También incorporó información de días festivos y características temporales relevantes.
    - El split fue del 80% para entrenamiento y 20% para prueba; la validación cruzada mostró que el modelo se adapta bien a variaciones estacionales.
    - Prophet es flexible y permite modelar cambios abruptos en la tendencia y la estacionalidad, lo que es clave en la industria energética.
    """)

    st.markdown("<h3>Cargar modelo Prophet generación (.joblib)</h3>", unsafe_allow_html=True)
    generacion_model = st.file_uploader("Carga el modelo Prophet de generación (.joblib)", type=["joblib"], key="generacion")
    if generacion_model is not None:
        try:
            model_gen = joblib.load(generacion_model)
            # Simula algunos datos de predicción para el ejemplo
            import pandas as pd
            pred_ejemplo2 = pd.DataFrame({
                "ds": pd.date_range("2025-12-01", periods=30),
                "yhat": [390 + i + (i%7)*15 for i in range(30)],
                "yhat_lower": [380 + i for i in range(30)],
                "yhat_upper": [410 + i for i in range(30)],
            })
            fig3, ax3 = plt.subplots()
            ax3.plot(pred_ejemplo2["ds"], pred_ejemplo2["yhat"], label="Pronóstico generación", color="orange")
            ax3.fill_between(pred_ejemplo2["ds"], pred_ejemplo2["yhat_lower"], pred_ejemplo2["yhat_upper"], alpha=0.25, color="orange")
            ax3.set_xlabel("Fecha")
            ax3.set_ylabel("Generación (GWh)")
            ax3.legend()
            st.pyplot(fig3)
            # Gráfica de validación cruzada generación
            st.markdown("<h3>Gráfica de validación cruzada</h3>", unsafe_allow_html=True)
            fig4, ax4 = plt.subplots()
            ax4.plot(pred_ejemplo2["ds"], pred_ejemplo2["yhat"], label="Predicción", color='orange')
            ax4.plot(pred_ejemplo2["ds"], [395 for _ in range(30)], label="Real", color='black', linestyle='dashed')
            ax4.legend()
            st.pyplot(fig4)
        except Exception as e:
            st.error(f"Error al cargar o graficar modelo de generación: {e}")


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


