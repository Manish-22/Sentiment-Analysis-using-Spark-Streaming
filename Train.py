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
            hashingTf=HashingTF(inputCol=tokenizer.getOutputCol(),outputCol="features",numFeatures=300)
            stringIndexer=StringIndexer(inputCol="sentiment", outputCol="label")



            pipeline=Pipeline(stages=[tokenizer,hashingTf,stringIndexer])
            pipelineFit=pipeline.fit(df)
            train_df=pipelineFit.transform(df)
            print(train_df.show(5))


            X = np.array(train_df.select('features').rdd.map(lambda x:x[0]).collect())
            print(X)
            print(X.shape)
            y = np.array(train_df.select('label').rdd.map(lambda x:x[0]).collect())

            
            lm_model.fit(X,y)
            predy=lm_model.predict(X)
            print(f"Logistic regression for Batch{iter}:")
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))
            print('\n\n')

            print(f"Stochaistic gradient descent for Batch{iter}:")
            sgd_model.partial_fit(X,y.ravel(), classes=[0.0,1.0])
            predy=sgd_model.predict(X)
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))
            print('\n\n')

            print(f"Multilayer Perceptron for Batch{iter}:")
            mlp_model.partial_fit(X,y.ravel(), classes=[0.0,1.0])
            predy=mlp_model.predict(X)
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))
            print('\n\n')

            print(f"Kmeans for Batch{iter}:")
            clus_model.partial_fit(X)
            predy=clus_model.predict(X)
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))
            print('\n\n')
            
            print(f"Multinomial Bayes{iter}:")
            mnb_model.partial_fit(X,y, classes=[0.0,1.0])
            predy=mnb_model.predict(X)
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))
            print('\n\n')

        except Exception as E:
            print('Somethings wrong I can feel it : ', E)
        
        iter+=1


iter =1
lm_model=lm.LogisticRegression(warm_start=True,maxIter=50,tol=1e-6) #high accuracy
sgd_model=SGDClassifier(alpha=0.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=True) #medium accuracy and medium precision
mlp_model=MLPClassifier(random_state=42, alpha=1e-5, hidden_layer_sizes=(5, 2)) #high recall
clus_model = MiniBatchKMeans(n_clusters=2, batch_size=1000, random_state=1)
mnb_model = MultinomialNB(alpha=0.0001, fit_prior=True, class_prior=None)
lines = lines.flatMap(lambda l: l.split('\\n",'))
lines = lines.map(lambda l:l[2:])
lines = lines.map(lambda l:l.split(',',1))		
lines.foreachRDD(rdd_test)


ssc.start()
ssc.awaitTermination(100000)
ssc.stop()



filename = 'lm_model.sav'
pickle.dump(lm_model, open(filename, 'wb'))

filename = 'sgd_model.sav'
pickle.dump(sgd_model, open(filename, 'wb'))

filename = 'mlp_model.sav'
pickle.dump(mlp_model, open(filename, 'wb'))

filename = 'clus_model.sav'
pickle.dump(clus_model, open(filename, 'wb'))

filename = 'mnb_model.sav'
pickle.dump(mnb_model, open(filename, 'wb'))