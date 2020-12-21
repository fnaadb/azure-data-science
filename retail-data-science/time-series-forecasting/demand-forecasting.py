# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC The objective of this notebook is to illustrate how we might generate a large number of fine-grained forecasts at the store-item level in an efficient manner leveraging the distributed computational power of Databricks. For this exercise, we will make use of an increasingly popular library for demand forecasting, FBProphet, which we will load into the notebook session associated with a cluster running Databricks 6.0 or higher. Please check out this blog post for more details. Please check out this https://databricks.com/blog/2020/01/27/time-series-forecasting-prophet-spark.html for more details.

# COMMAND ----------

# load fbprophet library
dbutils.library.installPyPI('FBProphet', version='0.5') # find latest version of fbprophet here: https://pypi.org/project/fbprophet/
dbutils.library.installPyPI('holidays','0.9.12') # this line is in response to this issue with fbprophet 0.5: https://github.com/facebook/prophet/issues/1293

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.types import *

# structure of the training data set
train_schema = StructType([
  StructField('date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('sales', IntegerType())
  ])

# read the training file into a dataframe
train = spark.read.csv(
  '/mnt/transactions/experimentdata/timeseries/input/train.csv', 
  header=True, 
  schema=train_schema
  )

# make the dataframe queriable as a temporary view
train.createOrReplaceTempView('train')

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from train

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from train limit 5

# COMMAND ----------

# MAGIC %sql
# MAGIC select year(date) as year, sum(sales) as sales from train group by year(date) order by year;

# COMMAND ----------

# MAGIC %md
# MAGIC t's very clear from the data that there is a generally upward trend in total unit sales across the stores. If we had better knowledge of the markets served by these stores, we might wish to identify whether there is a maximum growth capacity we'd expect to approach over the life of our forecast. But without that knowledge and by just quickly eyeballing this dataset, it feels safe to assume that if our goal is to make a forecast a few days, months or even a year out, we might expect continued linear growth over that time span.
# MAGIC 
# MAGIC Now let's examine seasonality. If we aggregate the data around the individual months in each year, a distinct yearly seasonal pattern is observed which seems to grow in scale with overall growth in sales:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY TRUNC(date,'MM')
# MAGIC ORDER BY month