import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Cargar la base de datos
df = pd.read_csv("riesgo_materno_limpio.csv")

print("Base de datos cargada correctamente")
print("Total de registros:", len(df))
print("Columnas:", df.columns.tolist())

# 2. Definir variables predictoras
variables = [
    "Edad",
    "Presion_Sistolica",
    "Presion_Diastolica",
    "Glucosa_Sangre",
    "Temperatura_Corporal",
    "Frecuencia_Cardiaca"
]

X = df[variables]

# Variable que queremos predecir
y = df["Nivel_Riesgo"]

# 3. Dividir los datos para entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Crear y entrenar Random Forest
modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

modelo.fit(X_train, y_train)

# 5. Evaluar el modelo
predicciones = modelo.predict(X_test)

print("\nRESULTADOS DEL MODELO")
print("Exactitud:", round(accuracy_score(y_test, predicciones) * 100, 2), "%")

print("\nREPORTE DE CLASIFICACIÓN")
print(classification_report(y_test, predicciones))

# 6. Guardar el modelo entrenado
joblib.dump(modelo, "modelo_riesgo_materno.pkl")

print("\nModelo guardado correctamente")