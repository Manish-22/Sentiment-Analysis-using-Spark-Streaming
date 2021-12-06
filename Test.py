import sys
import re

import pickle
import numpy as np
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
from pyspark.ml.feature import CountVectorizer, StopWordsRemover, Word2Vec, RegexTokenizer

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import matplotlib.pyplot as plt


sc=SparkContext.getOrCreate()
ssc = StreamingContext(sc, 5)
sqlContext = SQLContext(sc) #required to create dataframe





lines = ssc.socketTextStream("localhost", 6100)

def rdd_test(time,rdd):
    global iter, test_lm, test_sgd, test_mlp,test_mlp, test_mnb, test_clus
    print(f"===================={str(time)}====================")
	
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
            test_df=pipelineFit.transform(df)

            X = np.array(test_df.select('features').rdd.map(lambda x:x[0]).collect())
            y = np.array(test_df.select('label').rdd.map(lambda x:x[0]).collect())

            try:
                # loaded_lm_model.fit(X,y)
                predy=loaded_lm_model.predict(X)
                print(f"Logistic regression for Batch{iter}:")
                print("Accuracy:",accuracy_score(y, predy))
                print("Precision:",precision_score(y, predy))
                print("Recall:",recall_score(y, predy))
                print("Confusion Matrix:",confusion_matrix(y, predy))
                print('\n\n')
                test_lm+=accuracy_score(y,predy)
            except Exception as E:
                print('lm failed: ',E)
                print('\n\n')


            try:
                print(f"Stochaistic gradient descent for Batch{iter}:")
                # sgd_model.partial_fit(X,y, classes=[0.0,1.0])
                predy=loaded_sgd_model.predict(X)
                print("Accuracy:",accuracy_score(y, predy))
                print("Precision:",precision_score(y, predy))
                print("Recall:",recall_score(y, predy))
                print("Confusion Matrix:",confusion_matrix(y, predy))
                print('\n\n')
                test_sgd+=accuracy_score(y,predy)
            except Exception as E:
                print('sgd failed: ',E)
                print('\n\n')


            try:
                print(f"Multilayer Perceptron for Batch{iter}:")
                # mlp_model.partial_fit(X,y, classes=[0.0,1.0])
                predy=loaded_mlp_model.predict(X)
                print("Accuracy:",accuracy_score(y, predy))
                print("Precision:",precision_score(y, predy))
                print("Recall:",recall_score(y, predy))
                print("Confusion Matrix:",confusion_matrix(y, predy))
                print('\n\n')
                test_mlp+=accuracy_score(y,predy)

            except Exception as E:
                print('mlp failed: ',E)
                print('\n\n')



            try:
                print(f"Multinomial Bayes{iter}:")
                # mnb_model.partial_fit(X,y, classes=[0.0,1.0])
                predy=loaded_mnb_model.predict(X)
                print("Accuracy:",accuracy_score(y, predy))
                print("Precision:",precision_score(y, predy))
                print("Recall:",recall_score(y, predy))
                print("Confusion Matrix:",confusion_matrix(y, predy))
                print('\n\n')
                test_mnb+=accuracy_score(y,predy)

            except Exception as E:
                print('sgd failed: ',E)
                print('\n\n')

            try:
                print(f"Kmeans for Batch{iter}:")
                # clus_model.partial_fit(X)
                predy=loaded_clus_model.predict(X)
                print("Accuracy:",accuracy_score(y, predy))
                print("Precision:",precision_score(y, predy))
                print("Recall:",recall_score(y, predy))
                print("Confusion Matrix:",confusion_matrix(y, predy))
                print('\n\n')
                test_clus+=accuracy_score(y,predy)
                
            except Exception as E:
                print('sgd failed: ',E)
                print('\n\n')

            #print("===================Predictions===================")

            # accuracy = predictions.filter(predictions["label"] == predictions["prediction"]).count() / float(predictions.count())
            # auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
            # print("Accuracy Score: {0:.4f}".format(accuracy))
            # print("ROC-AUC: {0:.4f}".format(auc))

        except Exception as E:
            print('Somethings wrong I can feel it : ', E)

        iter+=1


test_lm=0
test_sgd=0
test_mlp=0
test_mnb=0
test_clus=0
iter=1


loaded_lm_model = pickle.load(open('lm_model.sav', 'rb'))
loaded_sgd_model = pickle.load(open('sgd_model.sav', 'rb'))
loaded_mlp_model = pickle.load(open('mlp_model.sav', 'rb'))
loaded_mnb_model = pickle.load(open('mnb_model.sav', 'rb'))
loaded_clus_model = pickle.load(open('clus_model.sav', 'rb'))


lines = lines.flatMap(lambda l: l.split('\\n",'))
lines = lines.map(lambda l:l[2:])
lines = lines.map(lambda l:l.split(',',1))		# split only by first ,
lines.foreachRDD(rdd_test)


ssc.start()
ssc.awaitTermination(60)
ssc.stop()


test_lm/=iter
test_sgd/=iter
test_mlp/=iter
test_mnb/=iter
test_clus/=iter

plt.bar(['LR','SGD','MLP','MNB','Kmeans'],[test_lm,test_sgd,test_mlp,test_mnb,test_clus])
plt.show()