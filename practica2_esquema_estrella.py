import pandas as pd
import numpy as np
import logging

# Configuración básica de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ejecutar_pipeline_etl():
    logging.info("Iniciando Pipeline ETL - Construcción de Esquema Estrella...")

    # =========================================================================
    # 0. SIMULACIÓN DE INGESTA OPERATIVA (OLTP)
    # =========================================================================
    np.random.seed(101)
    n = 500

    df_operativo = pd.DataFrame({
        'id_transaccion': range(1000, 1000 + n),
        'fecha_hora': pd.date_range(start='2026-01-01', periods=n, freq='h'),
        'cliente_nombre': np.random.choice(['Juan Pérez', 'María Gómez', 'Carlos Ruiz'], n),
        'cliente_ciudad': np.random.choice(['Puebla', 'CDMX', 'Guadalajara'], n),
        'producto_nombre': np.random.choice(['Laptop Dell', 'Mouse Logi', 'Monitor LG'], n),
        'categoria_prod': np.random.choice(['Cómputo', 'Accesorios', 'Pantallas'], n),
        'precio_unitario': np.random.choice([15000.0, 350.0, 4500.0], n),
        'cantidad': np.random.randint(1, 5, n)
    })
    logging.info(f"Ingesta OLTP completada: {len(df_operativo)} registros cargados.")

    # =========================================================================
    # 1. CONSTRUCCIÓN DE DIMENSIÓN CLIENTE (Dim_Cliente)
    # =========================================================================
    dim_cliente = df_operativo[['cliente_nombre', 'cliente_ciudad']].drop_duplicates().reset_index(drop=True)
    dim_cliente['id_cliente_sk'] = dim_cliente.index + 1  # Surrogate Key (SK)
    # Reordenar columnas para mantener SK primero
    dim_cliente = dim_cliente[['id_cliente_sk', 'cliente_nombre', 'cliente_ciudad']]

    # =========================================================================
    # 2. CONSTRUCCIÓN DE DIMENSIÓN PRODUCTO (Dim_Producto)
    # =========================================================================
    dim_producto = df_operativo[['producto_nombre', 'categoria_prod', 'precio_unitario']].drop_duplicates().reset_index(drop=True)
    dim_producto['id_producto_sk'] = dim_producto.index + 1  # Surrogate Key (SK)
    dim_producto = dim_producto[['id_producto_sk', 'producto_nombre', 'categoria_prod', 'precio_unitario']]

    # =========================================================================
    # 3. CONSTRUCCIÓN DE DIMENSIÓN TIEMPO (Dim_Tiempo) + EXTENSIONES
    # =========================================================================
    dim_tiempo = pd.DataFrame({'fecha_completa': df_operativo['fecha_hora'].dt.date.unique()})
    dim_tiempo['fecha_completa'] = pd.to_datetime(dim_tiempo['fecha_completa'])
    
    # Generación de Surrogate Key amigable para BI (AAAAMMDD)
    dim_tiempo['id_tiempo_sk'] = dim_tiempo['fecha_completa'].dt.strftime('%Y%m%d').astype(int)
    dim_tiempo['anio'] = dim_tiempo['fecha_completa'].dt.year
    dim_tiempo['mes'] = dim_tiempo['fecha_completa'].dt.month
    dim_tiempo['dia'] = dim_tiempo['fecha_completa'].dt.day
    
    # ATRIBUTOS ADICIONALES (Adecuación de la Práctica)
    dim_tiempo['dia_semana'] = dim_tiempo['fecha_completa'].dt.day_name()
    dim_tiempo['es_fin_semana'] = dim_tiempo['fecha_completa'].dt.dayofweek.isin([5, 6]).astype(int)
    
    dim_tiempo = dim_tiempo[['id_tiempo_sk', 'fecha_completa', 'anio', 'mes', 'dia', 'dia_semana', 'es_fin_semana']]

    # =========================================================================
    # 4. CONSTRUCCIÓN DE LA TABLA DE HECHOS (Fact_Ventas) + MÉTRICAS CALCULADAS
    # =========================================================================
    # Cruces (merges) para mapear Natural Keys a Surrogate Keys
    fact_ventas = df_operativo.merge(dim_cliente, on=['cliente_nombre', 'cliente_ciudad'])\
                              .merge(dim_producto, on=['producto_nombre', 'categoria_prod', 'precio_unitario'])

    fact_ventas['fecha_completa'] = pd.to_datetime(fact_ventas['fecha_hora'].dt.date)
    fact_ventas = fact_ventas.merge(dim_tiempo, on='fecha_completa')

    # Cálculos de Métricas
    fact_ventas['monto_total'] = fact_ventas['cantidad'] * fact_ventas['precio_unitario']
    fact_ventas['impuesto'] = np.round(fact_ventas['monto_total'] * 0.16, 2)  # IVA 16%
    fact_ventas['monto_final'] = fact_ventas['monto_total'] + fact_ventas['impuesto']

    # Selección final de Fact Table
    fact_table = fact_ventas[[
        'id_transaccion', 'id_cliente_sk', 'id_producto_sk', 'id_tiempo_sk', 
        'cantidad', 'monto_total', 'impuesto', 'monto_final'
    ]].sort_values(by='id_transaccion').reset_index(drop=True)

    # =========================================================================
    # 5. MOSTRAR RESULTADOS EN CONSOLA
    # =========================================================================
    print("\n==================================================")
    print("=== DIMENSIÓN CLIENTE (Dim_Cliente) ===")
    print("==================================================")
    print(dim_cliente.head())

    print("\n==================================================")
    print("=== DIMENSIÓN PRODUCTO (Dim_Producto) ===")
    print("==================================================")
    print(dim_producto.head())

    print("\n==================================================")
    print("=== DIMENSIÓN TIEMPO (Dim_Tiempo) ===")
    print("==================================================")
    print(dim_tiempo.head())

    print("\n==================================================")
    print("=== TABLA DE HECHOS (Fact_Ventas) ===")
    print("==================================================")
    print(fact_table.head())

    print("\n==================================================")
    print("=== RESUMEN ESTADÍSTICO FACT_VENTAS ===")
    print("==================================================")
    print(fact_table[['cantidad', 'monto_total', 'impuesto', 'monto_final']].describe().T)

    logging.info("Pipeline ETL completado exitosamente.")

if __name__ == "__main__":
    ejecutar_pipeline_etl()