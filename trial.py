import findspark
findspark.init()
import time
from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row,SQLContext
import sys
import requests
import json

conf=SparkConf()
conf.setAppName("BigData")
sc=SparkContext.getOrCreate(conf=conf)

sqlContext = SQLContext(sc)



ssc=StreamingContext(sc,5)
ssc.checkpoint("checkpoint_BIGDATA")

# RAW DATA
indata=ssc.socketTextStream("localhost",6100)
indata.pprint()

ssc.start()
ssc.awaitTermination(14)
ssc.stop()
