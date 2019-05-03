from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
from pyspark import SparkContext
from pyspark.sql import SQLContext

# Load training data
data = spark.read.csv(r'Ecommerce-Customers.csv',inferSchema=True,header=True)
data.show()
data.printSchema()


vectorAssembler = VectorAssembler(inputCols=['Avg Session Length','Time on App','Time on Website','Length of Membership'], outputCol='features')
output=vectorAssembler.transform(data)

final_data=output.select('features','Yearly Amount Spent')

final_data.show()

train_data,test_data = final_data.randomSplit([0.7,0.3])


train_data.describe().show()
# test = test_data.describe().show()
test = test_data.describe()

lr = LinearRegression(labelCol='Yearly Amount Spent')

lr_model=lr.fit(train_data)
test_results = lr_model.evaluate(test_data)

test_results.residuals.show()

test_results.rootMeanSquaredError
test_results.r2
final_data.describe().show()

unlabeled_data=test_data.select('features')
unlabeled_data.show()

Predictions=lr_model.transform(unlabeled_data)
Predictions.show()