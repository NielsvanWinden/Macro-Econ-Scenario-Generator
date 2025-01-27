from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("Multiple Linear Regression") \
    .getOrCreate()

# Sample data creation (you would replace this with your DataFrame)
data = [(1, 1.0, 2.0, 3.0, 4.0),
        (2, 2.0, 3.0, 4.0, 5.0),
        (3, 3.0, 4.0, 5.0, 6.0),
        (4, 4.0, 5.0, 6.0, 7.0),
        (5, 5.0, 6.0, 7.0, 8.0)]
columns = ["id", "feature1", "feature2", "feature3", "label"]

# Create a DataFrame
df = spark.createDataFrame(data, columns)

# Display the DataFrame
df.show()

# Prepare features (selecting multiple columns for the regression)
feature_columns = ["feature1", "feature2", "feature3"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Transform the DataFrame to have the features vector
df_transformed = assembler.transform(df).select("features", "label")
df_transformed.show()

# Split the data into training and testing sets
train_data, test_data = df_transformed.randomSplit([0.8, 0.2])

# Create a Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="label")

# Fit the model on the training data
lr_model = lr.fit(train_data)

# Print the coefficients and intercept for the linear regression model
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

# Make predictions on the test data
predictions = lr_model.transform(test_data)
predictions.show()

# Evaluate the model using RMSE (Root Mean Squared Error)
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Stop the Spark session
spark.stop()
