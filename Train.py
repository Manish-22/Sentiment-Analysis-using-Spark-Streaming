import sys
import numpy as np
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
from pyspark.ml.feature import CountVectorizer, StopWordsRemover, Word2Vec, RegexTokenizer
import pandas as pd
#spark=SparkSession.builder.appName("biG DATA").getOrCreate()
from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
import sys

from stream import sendPokemonBatchFileToSpark

sc=SparkContext.getOrCreate()
ssc = StreamingContext(sc, 5)
sqlContext = SQLContext(sc)

lines = ssc.socketTextStream("localhost", 6100)



def rdd_test(time,rdd):

    print(f"===================={str(time)}===============")
	
    if rdd.isEmpty():
        pass		
    else:
        try:
            df = sqlContext.createDataFrame(rdd, ['sentiment','message'])
            df= df.filter(df.sentiment != 'Sentiment')
            df.dropna()
            #df.show()

            regex = RegexTokenizer(inputCol= 'message' , outputCol= 'tokens', pattern= '\\W')
            remover2 = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
            stage_3 = Word2Vec(inputCol= 'filtered_words', outputCol= 'features', vectorSize= 100)
            indexer = StringIndexer(inputCol="sentiment", outputCol="label", stringOrderType='alphabetAsc')
            
            pipeline=Pipeline(stages=[regex, remover2, stage_3, indexer])
            pipelineFit=pipeline.fit(df)
            train_df=pipelineFit.transform(df)
            

            X= train_df.select(['features'])
            y= train_df.select(['label'])
            X.show()
            X = np.array(X.select('features').rdd.map(lambda x:x[0]).collect())
            y = np.array(y.select('label').rdd.map(lambda x:x[0]).collect())

            print(X,y,sep="\n")
            print(len(X),len(y),sep="\n")

        except Exception as E:
            print('Somethings wrong I can feel it : ', E)

lines = lines.flatMap(lambda l: l.split('\\n",'))
lines = lines.map(lambda l:l[2:])
lines = lines.map(lambda l:l.split(',',1))		# split only by first ,
lines.foreachRDD(rdd_test)


ssc.start()
ssc.awaitTermination(60)
ssc.stop()
