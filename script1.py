import os
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    when, col, avg, count, max, desc, to_timestamp, udf
)

from pyspark.sql.types import IntegerType


# Bloque comentado: Se deja como referencia de descarga, descompresión y carga en HDFS
"""
# 1. Descargar archivo original
print("Descargando dataset...")
os.system("wget https://physionet.org/files/mimic-iv-ed-demo/2.2/ed/vitalsign.csv.gz")

# 2. Descomprimir archivo
print("Descomprimiendo archivo...")
os.system("gunzip -f vitalsign.csv.gz")   # Esto genera vitalsign.csv

# 3. Subir archivo a HDFS en la carpeta Tarea3
print("Creando carpeta en HDFS (si no existe)...")
os.system("hdfs dfs -mkdir -p /Tarea3")

print("Subiendo archivo a HDFS...")
os.system("hdfs dfs -put -f vitalsign.csv /Tarea3")
"""

# 4. Inicializar sesión de Spark
print("Inicializando Spark...")
spark = SparkSession.builder.appName("Tarea3").getOrCreate()

# 5. Leer archivo desde HDFS
file_path = "hdfs://localhost:9000/Tarea3/vitalsign.csv"
print(f"Leyendo archivo desde {file_path} ...")
df = spark.read.format("csv").option("header","true").option("inferSchema","true").load(file_path)

# 6. Mostrar estructura del DataFrame
print("Esquema del dataframe:")
df.printSchema()

print("Proceso completado con éxito")

# Limpieza de datos: Conversión de tipos y manejo de valores vacíos
from pyspark.sql.functions import col, when, to_timestamp, round

df_clean = df \
    .withColumn("temperature",
               when(col("temperature") == "", None)
               .otherwise(col("temperature").cast("double"))) \
    .withColumn("heartrate",
               when(col("heartrate") == "", None)
               .otherwise(col("heartrate").cast("integer"))) \
    .withColumn("resprate",
               when(col("resprate") == "", None)
               .otherwise(col("resprate").cast("integer"))) \
    .withColumn("o2sat",
               when(col("o2sat") == "", None)
               .otherwise(col("o2sat").cast("integer"))) \
    .withColumn("sbp",
               when(col("sbp") == "", None)
               .otherwise(col("sbp").cast("integer"))) \
    .withColumn("dbp",
               when(col("dbp") == "", None)
               .otherwise(col("dbp").cast("integer"))) \
    .withColumn("charttime", to_timestamp(col("charttime"), "yyyy-MM-dd HH:mm:ss"))

# Redondeo de la temperatura a dos decimales
df_clean = df_clean.withColumn("temperature", round(col("temperature"), 2))


# Cálculo de Early Warning Score (EWS) basado en criterios clínicos
def calcular_ews(temperatura, fc, fr, sat_o2, sbp):
    score = 0

    # Frecuencia cardíaca
    if fc is not None:
        if fc >= 130: score += 3
        elif fc >= 120: score += 2
        elif fc >= 110: score += 1
        elif fc <= 50: score += 3
        elif fc <= 60: score += 1

    # Frecuencia respiratoria
    if fr is not None:
        if fr >= 25: score += 3
        elif fr >= 21: score += 2
        elif fr >= 12: score += 0
        elif fr <= 8: score += 3

    # Saturación de oxígeno
    if sat_o2 is not None:
        if sat_o2 <= 91: score += 3
        elif sat_o2 <= 93: score += 2
        elif sat_o2 <= 95: score += 1

    # Presión arterial sistólica
    if sbp is not None:
        if sbp <= 90: score += 3
        elif sbp <= 100: score += 2
        elif sbp <= 110: score += 1
        elif sbp >= 220: score += 3

    # Temperatura corporal
    if temperatura is not None:
        if temperatura <= 35.0: score += 3
        elif temperatura <= 36.0: score += 1
        elif temperatura >= 39.0: score += 2
        elif temperatura >= 38.1: score += 1

    return score

# Registrar la función como UDF para poder usarla en Spark
calcular_ews_udf = udf(calcular_ews, IntegerType())

# Aplicación de la columna score
df_con_alerta = df_clean.withColumn(
    "ews_score",
    calcular_ews_udf(col("temperature"), col("heartrate"),
                     col("resprate"), col("o2sat"), col("sbp"))
)

# Clasificación del nivel de riesgo según el puntaje
df_final = df_con_alerta.withColumn(
    "nivel_alerta",
    when(col("ews_score") >= 7, " CRÍTICO")
    .when(col("ews_score") >= 5, " ALTO")
    .when(col("ews_score") >= 3, " MODERADO")
    .otherwise(" ESTABLE")
)

# Mostrar pacientes con riesgo elevado
print("=== PACIENTES EN RIESGO ===")
df_final.filter(col("ews_score") >= 5).select(
    "subject_id", "charttime",
    round(col("ews_score"), 2).alias("ews_score"),
    "nivel_alerta",
    round(col("heartrate"), 2).alias("heartrate"),
    round(col("resprate"), 2).alias("resprate"),
    round(col("o2sat"), 2).alias("o2sat"),
    round(col("sbp"), 2).alias("sbp")
).orderBy(desc("ews_score")).show(20, False)

# Guardar resultados en formato Parquet
df_final.write.mode("overwrite").parquet("resultados_alertas")


# Estadísticas descriptivas básicas y distribución de alertas
print("\n=== ESTADÍSTICAS BÁSICAS ===")
df_final.select("heartrate", "resprate", "o2sat", "sbp", "ews_score") \
    .describe().show()

print("\n=== DISTRIBUCIÓN DE NIVELES DE ALERTA ===")
df_final.groupBy("nivel_alerta").count() \
    .orderBy(desc("count")).show()

print("\n=== TOP 10 PACIENTES MÁS CRÍTICOS ===")
df_final.groupBy("subject_id").agg(
    max("ews_score").alias("max_score"),
    avg("ews_score").alias("promedio_score"),
    count("ews_score").alias("total_registros")
).orderBy(desc("max_score"), desc("promedio_score")) \
 .show(10)
