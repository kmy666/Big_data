from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType

# Crear sesión de Spark
spark = SparkSession.builder \
    .appName("VitalsStreamingConsumer") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Definir el esquema EXACTO de los datos enviados por tu producer
schema = StructType([
    StructField("subject_id", IntegerType()),
    StructField("heartrate", IntegerType()),
    StructField("resprate", IntegerType()),
    StructField("o2sat", IntegerType()),
    StructField("sbp", IntegerType()),
    StructField("timestamp", IntegerType())
])

# Leer desde Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "sensor_vitalsign") \
    .load()

# Convertir JSON en columnas
parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Mostrar en consola los datos procesados
query = parsed_df \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
