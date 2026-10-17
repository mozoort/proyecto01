import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
# 1. GENERACIÓN DEL DATASET RUIDOSO SINTÉTICO
data = {
    'edad': [25, 30, np.nan, 45, 22, 120, 35, 28, np.nan, 40], # 120 es error/outlier
    'ingreso_mensual': [15000, 22000, 18000, np.nan, 12000, 950000, 25000, 21000, 19000, np.nan], # 950k es outlier
    'scoring_credito': [650, 700, 680, 710, 590, 800, 670, 640, 620, 730]
}

df_dirty = pd.DataFrame(data)

print("=== 1. DATASET RUIDOSO ORIGINAL ===")
print(df_dirty)

# 2. DETECCIÓN Y TRATAMIENTO DE OUTLIERS POR IQR
Q1 = df_dirty['ingreso_mensual'].quantile(0.25)
Q3 = df_dirty['ingreso_mensual'].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR

print(f"\n=== 2. DETECCIÓN ESTADÍSTICA DE OUTLIERS (IQR) ===")
print(f"Q1 (25%): ${Q1:,.2f}")
print(f"Q3 (75%): ${Q3:,.2f}")
print(f"IQR: ${IQR:,.2f}")
print(f"Límite Superior para Ingreso: ${limite_superior:,.2f}")

# Reemplazo de outliers/errores por NaN
df_prepared = df_dirty.copy()
df_prepared.loc[df_prepared['ingreso_mensual'] > limite_superior, 'ingreso_mensual'] = np.nan
df_prepared.loc[df_prepared['edad'] > 100, 'edad'] = np.nan # Corrección de Edad 120

print("\n=== DATASET PREPARADO (OUTLIERS Y ERRORES REEMPLAZADOS POR NaN) ===")
print(df_prepared)

# 3. IMPUTACIÓN MULTIVARIADA CON KNN IMPUTER (k=2)
imputer_knn2 = KNNImputer(n_neighbors=2)
df_clean_knn2 = pd.DataFrame(imputer_knn2.fit_transform(df_prepared), columns=df_prepared.columns)

print("\n=== 3. DATASET LIMPIO E IMPUTADO CON KNN (k=2) ===")
print(df_clean_knn2.round(2))

# 4. EXPERIMENTAL: VARIACIÓN DE VECINOS (k)
print("\n=== 4. EXPERIMENTO CON VARIACIÓN DE VECINOS (k) EN KNN ===")
for k in [1, 3, 5]:
    knn = KNNImputer(n_neighbors=k)
    df_k = pd.DataFrame(knn.fit_transform(df_prepared), columns=df_prepared.columns)
    print(f"\n--- Imputación con KNN (k={k}) ---")
    print(df_k[['edad', 'ingreso_mensual']].round(2))

# 5. COMPARATIVA: KNN vs MEDIA vs MEDIANA
imp_mean = SimpleImputer(strategy='mean')
imp_median = SimpleImputer(strategy='median')

df_mean = pd.DataFrame(imp_mean.fit_transform(df_prepared), columns=df_prepared.columns)
df_median = pd.DataFrame(imp_median.fit_transform(df_prepared), columns=df_prepared.columns)

print("\n=== 5. COMPARATIVA DE MÉTODOS DE IMPUTACIÓN ===")
comparativa = pd.DataFrame({
    'Edad_Original': df_prepared['edad'],
    'KNN_k2': df_clean_knn2['edad'],
    'Media': df_mean['edad'],
    'Mediana': df_median['edad'],
    'Ingreso_Original': df_prepared['ingreso_mensual'],
    'Ingreso_KNN_k2': df_clean_knn2['ingreso_mensual'],
    'Ingreso_Media': df_mean['ingreso_mensual'],
    'Ingreso_Mediana': df_median['ingreso_mensual']
})
print(comparativa.round(2))
