import findspark
findspark.init()
import time
from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row,SQLContext
import sys
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split

spark=SparkSession.builder.appName("bIG DATA").getOrCreate()

#Uncomment this for datastreaming
"""
# conf=SparkConf()
# conf.setAppName("BigData")
# sc=SparkContext.getOrCreate(conf=conf)

# ssc=StreamingContext(sc,5)

# dataStream=ssc.socketTextStream("localhost",6100)
# dataStream.
# ssc.start()
# ssc.awaitTermination()
# ssc.stop()
"""

lines = spark.readStream.format("socket").option("host", "localhost").option("port", 6100).load()
words = lines.select(
   explode(
       split(lines.value, " ")
   ).alias("word")
)
wordCounts = words.groupBy("word").count()
query = wordCounts \
    .writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()


