import io
import boto3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def pipeline_etl_completo(bucket_name="tu-bucket-real"):
    print("=== 1. FASE DE EXTRACCIÓN (Extract) ===")
    raw_data = {
        "cliente": ["A", "B", "C", "D"],
        "edad": [20, 40, 60, 30],
        "monto_compras": [1200.0, 50000.0, 30000.0, 15000.0],
    }
    df_raw = pd.DataFrame(raw_data)

    print("=== 2. FASE DE TRANSFORMACIÓN (Transform) ===")
    df_raw["categoria_edad"] = pd.cut(
        df_raw["edad"],
        bins=[0, 25, 50, 100],
        labels=["Joven", "Adulto", "Senior"],
    )

    scaler = MinMaxScaler()
    df_raw["monto_normalizado"] = scaler.fit_transform(
        df_raw[["monto_compras"]]
    )
    df_raw["score_promocional"] = df_raw["monto_normalizado"] * 100

    print("Dataset Transformado:")
    print(df_raw)

    print("\n=== 3. FASE DE CARGA (Load) ===")
    buffer = io.BytesIO()
    df_raw.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3_client = boto3.client("s3")
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key="processed/etl_output.parquet",
            Body=buffer.getvalue(),
        )
        print(
            "-> Éxito: Archivo Parquet comprimido cargado correctamente en "
            "la zona 'processed' de S3."
        )
    except Exception as e:
        print(f"Error al subir el archivo a S3: {e}")


if __name__ == "__main__":
    pipeline_etl_completo(bucket_name="mi-bucket-etl-14102026")