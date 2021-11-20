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

ssc=StreamingContext(sc,2)

dataStream=ssc.socketTextStream("localhost",6100)
dataStream.pprint()
ssc.start()
ssc.awaitTermination()
ssc.stop()
