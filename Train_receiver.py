import sys
import re

import pickle
from pyspark import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql import Row
from pyspark.streaming import StreamingContext
from pyspark.mllib.clustering import KMeans, KMeansModel, StreamingKMeans
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import operator
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import NGram, VectorAssembler
from pyspark.ml.feature import ChiSqSelector
from pyspark.sql.functions import when 
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import MultilayerPerceptronClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
#spark=SparkSession.builder.appName("biG DATA").getOrCreate()
from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
import sys

sc=SparkContext.getOrCreate()
sqlContext = SQLContext(sc)


df = sqlContext.read.option("header",True).csv('sentiment/smalltrain.csv')
#print(df.show(5))
df_train,df_test=df.randomSplit([0.8,0.2])
tokenizer = Tokenizer(inputCol="Tweet", outputCol="words")
hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "Sentiment", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])
pipelineFit = pipeline.fit(df)
train_df = pipelineFit.transform(df)
train_df=train_df.select("label","features")




lr= LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(train_df)

print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))
		




# lines = lines.flatMap(lambda l: l.split('\\n",'))
# lines = lines.map(lambda l:l[2:])
# lines = lines.map(lambda l:l.split(',',1))		# split only by first ,
# lines.foreachRDD(rdd_print)

# pickle.dump(model_multi, open("./model_multi.pkl", "wb"))

