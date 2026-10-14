import io
import logging
import boto3
import pandas as pd
from botocore.exceptions import ClientError

# Configuración básica de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cliente Boto3 para AWS S3[cite: 2, 5]
s3_client = boto3.client('s3', region_name='us-east-1')

# Nombre exacto de tu bucket
BUCKET_NAME = 'mi-bucket-etl-14102026'

def cargar_y_perfilar_parquet(file_key: str) -> pd.DataFrame:
    """Descarga un archivo Parquet desde S3 a memoria y perfila su contenido[cite: 2, 3, 5, 6]."""
    print(f"\n==========================================")
    print(f"=== Procesando objeto S3: {file_key} ===")
    print(f"==========================================")
    
    try:
        # Ingesta en memoria usando io.BytesIO[cite: 2, 3, 5, 6]
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
        body = response['Body'].read()
        df = pd.read_parquet(io.BytesIO(body))
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logging.error(f"Error S3 [{error_code}]: {e.response['Error']['Message']}")
        return None
    except Exception as e:
        logging.error(f"Error al procesar {file_key}: {str(e)}")
        return None

    # Perfilado de datos[cite: 3, 6]
    print(f"\n[+] Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    nulos = df.isnull().sum()
    print("\n[+] Conteo de Valores Nulos por Columna:")
    print(nulos[nulos > 0] if nulos.sum() > 0 else "Sin valores nulos.")

    print("\n[+] Perfil de Tipos de Datos (dtypes):")
    print(df.dtypes)

    print("\n[+] Muestra de los primeros registros (head):")
    print(df.head())

    print("\n[+] Resumen Estadístico Descriptivo:")
    print(df.describe().T)

    return df

if __name__ == "__main__":
    # Ingesta y perfilado del archivo Parquet en S3[cite: 3, 6]
    df_resultado = cargar_y_perfilar_parquet('processed/etl_output.parquet')