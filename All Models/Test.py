import sys
import pickle
import numpy as np
from pyspark import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql import Row
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import when 
from pyspark.ml.feature import RegexTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt


sc=SparkContext.getOrCreate()
ssc = StreamingContext(sc, 5)
sqlContext = SQLContext(sc) #required to create dataframe



lines = ssc.socketTextStream("localhost", 6100)

def rdd_test(time,rdd):
    global iter, test_lm, test_sgd, test_mlp,test_mlp,test_clus,test_mnb
    print(f"===================={str(time)}====================")
	
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

            predy=loaded_lm_model.predict(X)
            print(f"Logistic regression for Batch{iter}:")
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))
            print('\n\n')
            test_lm+=accuracy_score(y,predy)

            print(f"Stochaistic gradient descent for Batch{iter}:")
            predy=loaded_sgd_model.predict(X)
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))
            print('\n\n')
            test_sgd+=accuracy_score(y,predy)

            print(f"Multilayer Perceptron for Batch{iter}:")
            predy=loaded_mlp_model.predict(X)
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))
            print('\n\n')
            test_mlp+=accuracy_score(y,predy)

            #Multinomial Naive Bayes(MNB)
            print(f"Multinomial Bayes{iter}:")
            predy=loaded_mnb_model.predict(X)
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))
            print('\n\n')
            test_mnb+=accuracy_score(y,predy)

            #K-means(Unsupervised Learning)
            print(f"Kmeans for Batch{iter}:")
            predy=loaded_clus_model.predict(X)
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))
            print('\n\n')
            test_clus+=accuracy_score(y,predy)

            #print("===================Predictions===================")

            # accuracy = predictions.filter(predictions["label"] == predictions["prediction"]).count() / float(predictions.count())
            # auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
            # print("Accuracy Score: {0:.4f}".format(accuracy))
            # print("ROC-AUC: {0:.4f}".format(auc))

        except Exception as E:
            print('Somethings wrong I can feel it : ', E)

        iter+=1
        if iter>=320:
            sys.exit(0)


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

plt.bar(['LR','SGD','MLP','MNB','KMEANS'],[test_lm,test_sgd,test_mlp,test_mnb,test_clus])
plt.show()