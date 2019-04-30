from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

Spark = SparkSession.builder.appName('cluster').getOrCreate()

# Load training data
data = spark.read.csv(r'seeds_dataset.csv',inferSchema=True,header=True)
data.show()
data.printSchema()
data.head(1)
data.columns

vectorAssembler = VectorAssembler(inputCols=data.columns, outputCol='features')

final_data=vectorAssembler.transform(data)

final_data.printSchema()

scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures',)

Scaler_model = scaler.fit(final_data)
final_data = Scaler_model.transform(final_data)

final_data.head(1)

Kmeans = KMeans(featuresCol='scaledFeatures',k=3)

model = Kmeans.fit(final_data)
print('WSSSE')
print(model.computeCost(final_data))

centers = model.clusterCenters()
print(centers)
model.transform(final_data).show()
model.transform(final_data).select('prediction').show()