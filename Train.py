import sys
import numpy as np
import re

import pickle
from pyspark import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.mllib.clustering import KMeans, KMeansModel, StreamingKMeans
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from sklearn.linear_model import SGDClassifier
from pyspark.sql.functions import array
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from pyspark.ml.feature import CountVectorizer, StopWordsRemover, Word2Vec, RegexTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from pyspark import SparkContext
from pyspark.sql import SQLContext
import sklearn.linear_model as lm
from pyspark.sql.types import FloatType
import sys

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

            regex = RegexTokenizer(inputCol= 'message' , outputCol= 'tokens', pattern= '\\W')
            remover2 = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
            stage_3 = Word2Vec(inputCol= 'filtered_words', outputCol= 'features', vectorSize= 100)
            indexer = StringIndexer(inputCol="sentiment", outputCol="label", stringOrderType='alphabetAsc')
            
            pipeline=Pipeline(stages=[regex, remover2, stage_3, indexer])
            pipelineFit=pipeline.fit(df)
            train_df=pipelineFit.transform(df)
            
            X = np.array(train_df.select('features').rdd.map(lambda x:x[0]).collect())
            y = np.array(train_df.select('label').rdd.map(lambda x:x[0]).collect())

            
            model_lm.fit(X,y)
            predy=model_lm.predict(X)
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))


            model_sgd.partial_fit(X,y.ravel(), classes=[0.0,1.0])
            predy=model_sgd.predict(X)
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))


            
            model_mlp.partial_fit(X,y.ravel(), classes=[0.0,1.0])
            predy=model_mlp.predict(X)
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))

        except Exception as E:
            print('Somethings wrong I can feel it : ', E)

model_lm=lm.LogisticRegression(warm_start=True)
model_sgd=SGDClassifier(alpha=0.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=True)
model_mlp=MLPClassifier(random_state=1, max_iter=300)

lines = lines.flatMap(lambda l: l.split('\\n",'))
lines = lines.map(lambda l:l[2:])
lines = lines.map(lambda l:l.split(',',1))		
lines.foreachRDD(rdd_test)


ssc.start()
ssc.awaitTermination(60)
ssc.stop()
