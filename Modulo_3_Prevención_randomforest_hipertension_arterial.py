import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Cargar el archivo "Valores_Hipertension.csv" en un DataFrame
df = pd.read_csv("F:\\Utilidades\\Hello-Python-main\\Proyecto_3_prevención_hipertension\\Valores_Hipertension.csv")

# Convertir variables categóricas a numéricas
df['Genero'] = df['Genero'].map({'M': 0, 'F': 1})
df['HabitosFumador'] = df['HabitosFumador'].map({'No': 0, 'Sí': 1})
df['ActividadFisica'] = df['ActividadFisica'].map({'Sedentario': 0, 'Activo': 1})

# Dividir los datos en características (X) y etiquetas (y)
X = df.drop(['Nombre', 'Hipertension'], axis=1)
y = df['Hipertension']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo RandomForestClassifier con los mejores hiperparámetros encontrados
clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)
clf.fit(X_train, y_train)

# Obtener la importancia de cada característica
importancias = clf.feature_importances_

# Normalizar las importancias para que sumen 1 en total
importancias_normalizadas = importancias / np.sum(importancias)

# Imprimir la importancia de cada característica
print("Importancia de cada característica:")
for nombre, importancia in zip(X.columns, importancias_normalizadas):
    print(f"{nombre}: {importancia}")

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo en el conjunto de prueba
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo en el conjunto de prueba:", accuracy)

# Función para obtener los valores del nuevo paciente a través de inputs
def obtener_valores_paciente():
    """
    Esta función solicita al usuario ingresar información sobre el nuevo paciente.
    Se ingresan la edad, género, IMC, presión sistólica, presión diastólica, historial familiar,
    colesterol total, hábitos de fumador, actividad física y nivel de glucosa en sangre.
    Devuelve un diccionario con las características del paciente.
    """
    edad = int(input("Ingrese la edad del paciente: "))
    genero = input("Ingrese el género del paciente (M/F): ").upper()
    imc = float(input("Ingrese el IMC del paciente: "))
    presion_sistolica = int(input("Ingrese la presión sistólica del paciente: "))
    presion_diastolica = int(input("Ingrese la presión diastólica del paciente: "))
    historial_familiar = int(input("Ingrese el historial familiar de hipertensión (0 - No, 1 - Sí): "))
    colesterol_total = int(input("Ingrese el colesterol total del paciente: "))
    habitos_fumador = input("Ingrese los hábitos de fumador del paciente (No/Sí): ").title()
    actividad_fisica = input("Ingrese la actividad física del paciente (Sedentario/Activo): ").title()
    glucosa_sangre = int(input("Ingrese el nivel de glucosa en sangre del paciente: "))

    # Convertir los valores categóricos a numéricos usando los mismos mapeos que en el DataFrame original
    genero_num = 0 if genero == 'M' else 1
    habitos_fumador_num = 0 if habitos_fumador == 'No' else 1
    actividad_fisica_num = 0 if actividad_fisica == 'Sedentario' else 1

    return {
        'Edad': edad,
        'Genero': genero_num,
        'IMC': imc,
        'PresionSistolica': presion_sistolica,
        'PresionDiastolica': presion_diastolica,
        'HistorialFamiliar': historial_familiar,
        'ColesterolTotal': colesterol_total,
        'HabitosFumador': habitos_fumador_num,
        'ActividadFisica': actividad_fisica_num,
        'GlucosaEnSangre': glucosa_sangre
    }

# Obtener los valores del nuevo paciente
nuevo_paciente = obtener_valores_paciente()
nuevo_paciente_df = pd.DataFrame([nuevo_paciente])

# Obtener la importancia probabilística de cada característica para el nuevo paciente
importancia_probabilistica = 0
for nombre, valor in nuevo_paciente.items():
    idx = X.columns.get_loc(nombre)
    importancia_probabilistica += importancias_normalizadas[idx] * valor

# Realizar la predicción para el nuevo paciente
prediccion = clf.predict(nuevo_paciente_df)

# Realizar la predicción para el nuevo paciente
if importancia_probabilistica >= 0.5:
    print("El paciente tiene posibilidad de tener hipertensión arterial en este momento.")
else:
    print("El paciente no tiene posibilidad de tener hipertensión arterial en este momento.")