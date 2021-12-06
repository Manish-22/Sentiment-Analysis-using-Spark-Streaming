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
from sklearn.neural_network import MLPClassifier
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from pyspark import SparkContext
from pyspark.sql import SQLContext
import sklearn.linear_model as lm
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import FloatType
import sys
from pyspark.ml.feature import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from pyspark.ml.feature import HashingTF,IDF,Tokenizer,StringIndexer
from pyspark.ml.classification import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

sc=SparkContext.getOrCreate()
ssc = StreamingContext(sc, 5)
sqlContext = SQLContext(sc)

lines = ssc.socketTextStream("localhost", 6100)



def rdd_test(time,rdd):
    global iter
    print(f"===================={str(time)}===============")
	
    if rdd.isEmpty():
        pass		
    else:
        try:
            df = sqlContext.createDataFrame(rdd, ['sentiment','message'])
            df= df.filter(df.sentiment != 'Sentiment')
            df.dropna()


            tokenizer = RegexTokenizer(inputCol= 'message' , outputCol= 'tokens', pattern= '\\W')
            hashingTf=HashingTF(inputCol=tokenizer.getOutputCol(),outputCol="features",numFeatures=1000)
            stringIndexer=StringIndexer(inputCol="sentiment", outputCol="label")



            pipeline=Pipeline(stages=[tokenizer,hashingTf,stringIndexer])
            pipelineFit=pipeline.fit(df)
            train_df=pipelineFit.transform(df)
            print(train_df.show(5))


            X = np.array(train_df.select('features').rdd.map(lambda x:x[0]).collect())
            y = np.array(train_df.select('label').rdd.map(lambda x:x[0]).collect())


            print(f"Stochaistic gradient descent for Batch{iter}:")
            sgd_model.partial_fit(X,y.ravel(), classes=[0.0,1.0])
            predy=sgd_model.predict(X)
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))
            print('\n\n')


        except Exception as E:
            print('Somethings wrong I can feel it : ', E)
        
        iter+=1


iter =1


lines = lines.flatMap(lambda l: l.split('\\n",'))
lines = lines.map(lambda l:l[2:])
lines = lines.map(lambda l:l.split(',',1))


sgd_model=SGDClassifier()

lines.foreachRDD(rdd_test)



ssc.start()
ssc.awaitTermination(100000)
ssc.stop()



# filename = 'lm_model.sav'
# pickle.dump(lm_model, open(filename, 'wb'))

filename = 'sgd_model.sav'
pickle.dump(sgd_model, open(filename, 'wb'))

# filename = 'mlp_model.sav'
# pickle.dump(mlp_model, open(filename, 'wb'))