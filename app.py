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
    - Se utilizó Prophet para pronosticar la demanda y generación eléctrica diaria en Colombia.
    - Los festivos se incluyeron como variables importantes en la ingeniería de características.
    - El set de datos se dividió en un 80% para entrenamiento y 20% para prueba.
    - La validación cruzada manual permitió medir la robustez del modelo y evitar el sobreajuste.
    - Prophet descompone las series temporales en tendencia, estacionalidad y efectos especiales para mejorar la interpretación y precisión del pronóstico energético.
    """)

    # Sección modelo demanda
    st.markdown("<h2>Modelo demanda</h2>", unsafe_allow_html=True)
    try:
        st.image("demanda.png", caption='Demanda: Real vs Predicho (con Validación Cruzada)', use_column_width=True)
    except Exception:
        st.warning("No se pudo cargar la gráfica demanda.png. Verifica que esté en la raíz con app.py.")
    st.markdown("R² global (manual, cross-validation): **0.2162**")

    # Sección modelo generación
    st.markdown("<h2>Modelo generación</h2>", unsafe_allow_html=True)
    try:
        st.image("generacion.png", caption='Generación: Real vs Predicho (con Validación Cruzada)', use_column_width=True)
    except Exception:
        st.warning("No se pudo cargar la gráfica generacion.png. Verifica que esté en la raíz con app.py.")
    st.markdown("R² global (manual, cross-validation): **0.7655**")



# 5. CALIFICACIÓN DE MODELOS
elif selected_section == "Calificación de modelos":
    st.markdown("""
## Calificación de los modelos creados

Los dos modelos creados —demanda y generación energética con Prophet— pueden calificarse de la siguiente manera para uso analítico:

***

### Modelo de demanda energética

**Calificación:**  
- **Adecuado para análisis exploratorio, monitoreo tendencial y escenarios base.**
- **Efectividad:**  
  - Tras la limpieza de outliers extremos, el modelo muestra un desempeño aceptable, con un R² positivo (>0.21), bajos errores relativos (MAPE y SMAPE <8%) y errores absolutos (MAE y RMSE) razonables.
  - Es útil para proyecciones, validaciones de hipótesis y planificación, pero se recomienda complementar con variables exógenas o métodos más avanzados si se requiere alto nivel de confiabilidad o segmentación detallada[1][2].
- **Limitaciones:**  
  - La historia disponible es corta, así que los patrones de largo plazo pueden estar subrepresentados.
  - Los eventos fortuitos o cambios bruscos futuros pueden no estar del todo anticipados.

***

### Modelo de generación energética

**Calificación:**  
- **Muy robusto y altamente recomendado para análisis predictivo y comparativo.**
- **Efectividad:**  
  - El modelo retiene un R² cruzado muy alto (~0.77), una baja dispersión de errores y excelente ajuste visual en la comparación real vs. predicho.
  - Es confiable para reportes de largo plazo, benchmarking y decisiones de política energética.
- **Limitaciones:**  
  - Como todos los modelos Prophet, depende de que el futuro se comporte similar al pasado (sin eventos exógenos o rupturas drásticas). Si se prevén cambios estructurales (nuevas fuentes o apagones sistémicos), conviene agregar regresores o capas de modelado.

***

## Resumen de calificación

| Variable      | Uso Analítico            | Robustez | Nivel recomendación |
|---------------|-------------------------|----------|---------------------|
| Demanda       | Exploración, tendencia   | Aceptable| Bueno               |
| Generación    | Predicción y análisis    | Alto     | Muy Bueno           |

Ambos modelos son útiles para análisis y toma de decisiones, pero el de generación ofrece resultados más robustos y estables; el de demanda es satisfactorio para escenarios base y monitoreo operativo tras la mejora en calidad de datos[1][2].

Fuentes  
[1] prophet: Automatic Forecasting Procedure https://community.r-multiverse.org/prophet/prophet.pdf  
[2] Time Series Forecasting using Facebook Prophet https://www.nileshdalvi.com/blog/time-series-prophet/
    """)


# 6. COMPARACIÓN Y CONCLUSIONES
elif selected_section == "Comparación y conclusiones":

    st.markdown("<h1>Comparación generación vs demanda y conclusiones</h1>", unsafe_allow_html=True)
    st.write("Abajo se muestra la comparación entre la generación y la demanda eléctrica de Colombia:")

    
    try:
        st.image("comparacion.png", caption='Demanda vs Generacion)', use_column_width=True)
    except Exception:
        st.warning("No se pudo cargar la gráfica comparacion.png. Verifica que esté en la raíz con app.py.")
    
    st.markdown("""
# Validación de conclusiones
- Según la Unidad de Planeación Minero Energética (UPME), la demanda eléctrica en Colombia crecerá en promedio 2,38% anual hasta 2038, lo que pone en tensión la infraestructura existente y podría generar un déficit estructural de energía a partir de 2027 si no se realizan nuevas inversiones en generación.
- A la fecha, Colombia debería agregar entre 3.000 y 4.000 MW de capacidad firme anual para satisfacer el crecimiento de la demanda, pero actualmente solo se está alcanzando cerca del 30% de ese objetivo.
- El potencial de crecimiento de energías renovables como la solar y eólica es significativo. En los últimos tres años, la capacidad solar en operación comercial ha crecido hasta un 187%, impulsada por políticas nacionales y el Plan Nacional de Desarrollo (PND).
- Ya se han adjudicado proyectos por más de 4.500 MW en fuentes no convencionales de energía renovable (FNCER).
- La meta oficial del país es duplicar la capacidad instalada en energías renovables para 2030 y alcanzar al menos 6 GW para 2026.
- Sin embargo, existen dificultades estructurales, incluyendo retrasos en licenciamiento ambiental, consultas previas, y limitaciones en infraestructura de transmisión, que frenan la entrada de nuevos proyectos, especialmente eólicos y solares.

## Factores de variación del crecimiento de la generación renovable
- El crecimiento real de la capacidad instalada en renovables depende de factores como:
    - Agilidad regulatoria y administrativa en la aprobación de proyectos.
    - Inversiones en infraestructura de transmisión eléctrica y sistemas de almacenamiento.
    - Adopción masiva de generación distribuida (paneles solares residenciales e industriales).
    - Estabilidad y atractivo de los esquemas de contratos de compra de energía (PPA).
    - Políticas de incentivo a autogeneración y comunidades energéticas.
    - En contextos de alta penetración renovable, la producción de energía puede no coincidir con los picos de consumo diario, exigiendo una gestión eficiente de los excedentes (baterías, sistemas de almacenamiento).

## Informe sobre proyecciones y factores de crecimiento de la generación eléctrica en Colombia (2025)
		
La demanda eléctrica en Colombia aumentará en promedio 2,38% anual en los próximos años, presionando la infraestructura actual y exigiendo inversiones para evitar déficit a partir de 2027. Aunque la capacidad instalada de energía renovable (solar y eólica) está creciendo rápidamente y existen metas de duplicación para 2030, la velocidad real del crecimiento depende de factores regulatorios, de infraestructura y financieros. Los retrasos en licenciamiento ambiental, limitaciones en transmisión eléctrica y la necesidad de facilitar la autogeneración y comunidades energéticas pueden ralentizar la expansión. Si bien Colombia ha avanzado en renovables (más de 4.500 MW adjudicados y el 14% de la matriz ya renovable), mantener la sostenibilidad energética requerirá continuar promoviendo inversiones, evitar cuellos de botella, y fortalecer la gestión de almacenamiento y contratos de largo plazo para mitigar la volatilidad de precios y demanda.

Por ello, los modelos de proyección de demanda y generación deben considerar las incertidumbres y la variabilidad de crecimiento en energías renovables, sobre todo en la expansión de paneles solares y plantas eólicas, además del ritmo de adopción y factores socioeconómicos.
    """)




# 7. FUENTES
elif selected_section == "Fuentes":
    
    st.markdown("""
## Fuentes:
- UPME (Unidad de Planeación Minero Energética): Proyecciones oficiales de demanda y generación.
- Atlas Renewable Energy: Perspectivas y tendencias del sector energético colombiano.
- El Colombiano: Entrevista y análisis sobre el crecimiento y déficit del sector eléctrico.
- Caracol Radio: Informe sobre déficit energético proyectado para 2027.
- SITTCA (Sistema de Información Técnica y Comercial de Colombia): Análisis de impulso renovable y cifras recientes.
- DNP (Departamento Nacional de Planeación): Reportes y proyecciones de energía renovable.
- El Universal: Artículo sobre riesgos de desabastecimiento energético para 2026.
- SER Colombia: Documento oficial sobre expansión de fuentes no convencionales de energía renovable (FNCER).
- Invest in Colombia: Datos sobre adjudicación y crecimiento en proyectos renovables.
- SEI (Stockholm Environment Institute): Estudios sobre energía solar, eólica y comunidades energéticas en Colombia.
- Climatetracker Latinoamérica: Análisis de los desafíos y oportunidades de las renovables frente al desabastecimiento'''


st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por equipo frozado")


