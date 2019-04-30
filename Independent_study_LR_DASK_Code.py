#reading the csv files
import dask.dataframe as dd
df = dd.read_csv('blackfriday_train.csv')
test=dd.read_csv("blackfriday_test.csv")

#having a look at the head of the dataset
df.head()

#finding the null values in the dataset
df.isnull().sum().compute()

#defining the data and target
categorical_variables = df[['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']]
target = df['Purchase']

#creating dummies for the categorical variables
data = dd.get_dummies(categorical_variables.categorize()).compute()

#converting dataframe to array
datanew=data.values

#fit the model
from dask_ml.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(datanew, target)

#preparing the test data
test_categorical = test[['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']]
test_dummy = dd.get_dummies(test_categorical.categorize()).compute()
testnew = test_dummy.values

#predict on test and upload
pred=lr.predict(testnew)


#Clustering/K-Means
from dask_ml.cluster import KMeans
model = KMeans()
model.fit(datanew, target)
