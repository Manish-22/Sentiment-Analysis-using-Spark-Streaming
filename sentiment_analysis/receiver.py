import sys
import re

from pyspark import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql import Row
from pyspark.streaming import StreamingContext
from pyspark.mllib.clustering import KMeans, KMeansModel, StreamingKMeans
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import operator

#spark=SparkSession.builder.appName("biG DATA").getOrCreate()

sc=SparkContext.getOrCreate()

ssc = StreamingContext(sc, 5)

sqlContext = SQLContext(sc)


def streamrdd_to_df(srdd):
    sdf = sqlContext.createDataFrame(srdd)
    sdf.show(n=2, truncate=False)
    return sdf



lines = ssc.socketTextStream("localhost", 6100)


def rdd_print(time,rdd):
	
	print(f"===================={str(time)}===============")
	b = rdd.flatMap(lambda l: l.split('\\n",'))
	a = b.map(lambda l:l[2:])
	a = a.map(lambda l:l.split(',',2))		# split only by first ,
	
	if a.isEmpty():
		pass
	
	
		
	else:
		
#	
		try:
			df = sqlContext.createDataFrame(a, ['col1','col2'])
			df.show(20)
		except:
			print('Not working')
			pass
	
	

lines.foreachRDD(rdd_print)
#lines.pprint()


ssc.start()
ssc.awaitTermination()
ssc.stop()

