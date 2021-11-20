import findspark
findspark.init()
import time
from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row,SQLContext
import sys
import requests

conf=SparkConf()
conf.setAppName("BigData")
sc=SparkContext.getOrCreate(conf=conf)

ssc=StreamingContext(sc,5)

dataStream=ssc.socketTextStream("localhost",6100)
ssc.start()
ssc.awaitTermination()
ssc.stop()
