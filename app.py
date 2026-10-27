import streamlit as st
import pandas as pd
import joblib
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_riesgo_materno.pkl")

modelo = cargar_modelo()
st.set_page_config(
    page_title="Datos que protegen dos vidas",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Datos que protegen dos vidas")
st.subheader("Análisis del riesgo materno mediante inteligencia analítica")

st.info(
    "Herramienta académica de análisis de datos. "
    "No sustituye la valoración de un profesional de salud."
)

# Cargar la base de datos
try:
    df = pd.read_csv("riesgo_materno_limpio.csv")

    st.success("Base de datos cargada correctamente.")

    st.metric("Total de registros", len(df))

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(10), use_container_width=True)

except Exception as e:
    st.error(f"Error al cargar los datos: {e}")


    # ==========================================
# EVALUACIÓN INTERACTIVA DEL RIESGO MATERNO
# ==========================================

st.divider()

st.header("🩺 Evaluación interactiva del riesgo materno")

st.info(
    "Ingrese los datos para consultar referencias clínicas "
    "orientativas. Esta herramienta no reemplaza la valoración médica."
)

st.subheader("📋 Datos de la paciente")

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input(
        "Edad (años)",
        min_value=10,
        max_value=100,
        value=30
    )

    sistolica = st.number_input(
        "Presión arterial sistólica (mmHg)",
        min_value=50,
        max_value=250,
        value=120
    )

    diastolica = st.number_input(
        "Presión arterial diastólica (mmHg)",
        min_value=30,
        max_value=150,
        value=80
    )

with col2:
    glucosa = st.number_input(
        "Glucosa sanguínea (unidad de la base por verificar)",
        min_value=0.0,
        max_value=30.0,
        value=7.0
    )

    temperatura = st.number_input(
        "Temperatura corporal (°F)",
        min_value=90.0,
        max_value=110.0,
        value=98.0
    )

    frecuencia = st.number_input(
        "Frecuencia cardiaca (latidos/minuto)",
        min_value=30,
        max_value=220,
        value=80
    )

st.caption(
    "La temperatura se registra en °F para mantener "
    "la unidad utilizada en la base de datos."
)

temperatura_c = (temperatura - 32) * 5 / 9

st.write(
    f"🌡️ Temperatura equivalente: {temperatura_c:.1f} °C"
)

# ==========================================
# PREDICCION INTERACTIVA DEL RIESGO MATERNO
# ==========================================

st.subheader("🤖 Predicción del riesgo materno con IA")

if st.button("🔍 Predecir riesgo materno"):

    datos_paciente = pd.DataFrame([{
        "Edad": edad,
        "Presion_Sistolica": sistolica,
        "Presion_Diastolica": diastolica,
        "Glucosa_Sangre": glucosa,
        "Temperatura_Corporal": temperatura,
        "Frecuencia_Cardiaca": frecuencia
    }])

    prediccion = modelo.predict(datos_paciente)[0]

    st.write("### Resultado del modelo Random Forest")

    if prediccion == "Riesgo alto":
        st.error("🔴 Clasificación estimada: RIESGO ALTO")

    elif prediccion == "Riesgo medio":
        st.warning("🟡 Clasificación estimada: RIESGO MEDIO")

    elif prediccion == "Riesgo bajo":
        st.success("🟢 Clasificación estimada: RIESGO BAJO")

    else:
        st.info(f"Clasificación obtenida: {prediccion}")

    st.caption(
    "Resultado generado mediante inteligencia artificial "
    "con fines académicos. El modelo no está validado "
    "para diagnóstico clínico ni para orientar decisiones médicas."
)

if st.button("🔍 Consultar referencias", type="primary"):
    st.subheader("📊 Resultados de la consulta")

    st.write(f"Edad: {edad} años")
    st.write(f"Presión arterial: {sistolica}/{diastolica} mmHg")
    st.write(f"Glucosa registrada: {glucosa}")
    st.write(f"Temperatura: {temperatura_c:.1f} °C")
    st.write(f"Frecuencia cardiaca: {frecuencia} latidos/minuto")

  
    st.warning(
        "Esta primera versión muestra los datos ingresados. "
        "La clasificación de riesgo bajo, medio o alto "
        "se habilitará cuando integremos y validemos "
        "el modelo Random Forest."
    )

    # ==========================================
# TABLA DE REFERENCIAS CLÍNICAS ORIENTATIVAS
# ==========================================

st.divider()

st.header("📚 Valores de referencia maternos")

st.info(
    "Los siguientes valores son orientativos. "
    "Su interpretación depende de la edad gestacional, "
    "los antecedentes y la evaluación clínica."
)

referencias = pd.DataFrame({
    "Variable": [
        "Edad materna",
        "Presión sistólica",
        "Presión diastólica",
        "Glucosa sanguínea",
        "Temperatura corporal",
        "Frecuencia cardiaca"
    ],
    "Referencia orientativa": [
        "No existe una edad única normal",
        "Inferior a 140 mmHg",
        "Inferior a 90 mmHg",
        "70–99 mg/dL (3,9–5,5 mmol/L)",
        "Aproximadamente 36,1 a 37,2 °C",
        "60–100 latidos/minuto"
    ],
    "Observaciones": [
        "A partir de 35 años aumenta el riesgo de ciertas complicaciones",
        "140 mmHg o más requiere evaluación",
        "90 mmHg o más requiere evaluación",
        "Verificar unidades y si la medición fue en ayunas",
        "38 °C o más puede indicar fiebre",
        "Interpretar según edad gestacional y contexto clínico"
    ]
})

st.dataframe(
    referencias,
    hide_index=True,
    width="stretch"
)

st.caption(
    "Referencias educativas. No constituyen criterios "
    "diagnósticos ni sustituyen una valoración médica."
)