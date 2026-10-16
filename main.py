import io
import json
import logging
import boto3
import pandas as pd
from botocore.exceptions import ClientError

# Configuración básica de logs corregida
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 1. Configuración de cliente Boto3 para conexión con AWS Academy S3
s3_client = boto3.client('s3', region_name='us-east-1')

# Nombre exacto de tu bucket
BUCKET_NAME = 'mi-bucket-etl-14102026'

def cargar_y_perfilar_s3(file_key: str, is_json: bool = False, is_parquet: bool = False) -> pd.DataFrame:
    """
    Descarga archivos desde S3 directamente a memoria (RAM) usando io.BytesIO
    y genera un perfilamiento de datos completo. Maneja excepciones de conexion.
    """
    print(f"\n==================================================")
    print(f"=== Procesando objeto S3: {file_key} ===")
    print(f"==================================================")
    
    try:
        # Ingesta en memoria RAM (sin escribir en el disco local)
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
        body = response['Body'].read()

        if is_json:
            data_json = json.loads(body.decode('utf-8'))
            df = pd.json_normalize(data_json)
        elif is_parquet:
            df = pd.read_parquet(io.BytesIO(body))
        else:
            df = pd.read_csv(io.BytesIO(body))

    except ClientError as e:
        error_code = e.response['Error']['Code']
        logging.error(f"Error de acceso a S3 [{error_code}]: {e.response['Error']['Message']}")
        return None
    except Exception as e:
        logging.error(f"Error procesando el archivo {file_key}: {str(e)}")
        return None

    # Perfilamiento de datos e identificación de esquemas
    print(f"\n[+] Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    print("\n[+] Perfil de Tipos de Datos (dtypes):")
    print(df.dtypes)

    print("\n[+] Conteo de Valores Nulos por Columna:")
    nulos = df.isnull().sum()
    print(nulos if nulos.sum() > 0 else "Sin valores nulos.")

    print("\n[+] Muestra de los primeros registros (head):")
    print(df.head())

    print("\n[+] Resumen de Estadísticos Descriptivos (Cuantitativos):")
    print(df.describe().T)

    return df

if __name__ == "__main__":
    # Ingesta y perfilamiento del archivo Parquet almacenado en tu S3
    df_parquet = cargar_y_perfilar_s3('processed/etl_output.parquet', is_parquet=True)
