from faker import Faker
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. CONFIGURACIÓN DE SEMILLAS ALEATORIAS (REPRODUCIBILIDAD)
fake = Faker('es_MX')
Faker.seed(42)
np.random.seed(42)

# 2. GENERACIÓN PROGRAMÁTICA DE DATASET SINTÉTICO (1,000 REGISTROS)
registros = []
for i in range(1000):
    registros.append({
        'id_cliente': i + 1,
        'nombre': fake.name(),
        'email': fake.email(),
        'direccion': fake.address().replace('\n', ', '), # Variable sintética adicional 1
        'telefono': fake.phone_number(),                 # Variable sintética adicional 2
        'estado_civil': np.random.choice(['Soltero', 'Casado', 'Divorciado'], p=[0.5, 0.4, 0.1]),
        'ingreso_anual': round(np.random.uniform(100000, 1000000), 2),
        'churn': np.random.choice([0, 1], p=[0.8, 0.2]) # 20% tasa de abandono desbalanceada
    })

df_sintetico = pd.DataFrame(registros)

print("=== 1. MUESTRA DEL DATASET SINTÉTICO GENERADO CON FAKER ===")
print(df_sintetico.head(4))

# 3. CODIFICACIÓN CATEGÓRICA (ONE-HOT ENCODING)
df_encoded = pd.get_dummies(df_sintetico, columns=['estado_civil'], drop_first=True, dtype=int)

print("\n=== 2. DATASET TRAS CODIFICACIÓN CATEGÓRICA ===")
print(df_encoded.head(4)[['ingreso_anual', 'estado_civil_Divorciado', 'estado_civil_Soltero']])

# 4. SELECCIÓN DE CARACTERÍSTICAS (X) Y TARGET (y)
X = df_encoded.drop(columns=['id_cliente', 'nombre', 'email', 'direccion', 'telefono', 'churn'])
y = df_encoded['churn']

# 5. PARTICIÓN ESTRATIFICADA (TRAIN 80% / TEST 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\n=== 3. RESULTADOS DE LA PARTICIÓN ESTRATIFICADA (TRAIN 80% / TEST 20%) ===")
print(f"Dimensiones Dataset Completo: {X.shape} | Proporción Churn Global: {y.mean()*100:.2f}%")
print(f"Dimensiones Train Set:        {X_train.shape} | Proporción Churn Train:  {y_train.mean()*100:.2f}%")
print(f"Dimensiones Test Set:         {X_test.shape} | Proporción Churn Test:   {y_test.mean()*100:.2f}%")

# 6. TABLA COMPARATIVA DE PROPORCIONES DE CHURN
tabla_resumen = pd.DataFrame({
    'Conjunto': ['Completo', 'Train (80%)', 'Test (20%)'],
    'Total_Registros': [len(y), len(y_train), len(y_test)],
    'Total_Churn_1': [y.sum(), y_train.sum(), y_test.sum()],
    'Proporcion_Churn_%': [round(y.mean()*100, 2), round(y_train.mean()*100, 2), round(y_test.mean()*100, 2)]
})

print("\n=== TABLA COMPARATIVA DE PROPORCIÓN DE CHURN ===")
print(tabla_resumen)
