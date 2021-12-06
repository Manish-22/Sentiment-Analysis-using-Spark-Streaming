import numpy as np
import pickle
from pyspark import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline, param
from pyspark.sql.functions import array
from pyspark.ml.feature import RegexTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from pyspark import SparkContext
from pyspark.sql import SQLContext
import sklearn.linear_model as lm
from pyspark.sql.types import FloatType
import sys
from sklearn.model_selection import GridSearchCV
from pyspark.ml.feature import HashingTF,StringIndexer
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
            y = np.array(train_df.select('label').rdd.map(lambda x:x[0]).collect())

            
            #clf.fit(X,y)
            #print(clf.cv_results_)
            lm_model.fit(X,y)
            predy=lm_model.predict(X)
            print(f"Logistic regression for Batch{iter}:")
            print("Accuracy:",accuracy_score(y, predy))
            print("Precision:",precision_score(y, predy))
            print("Recall:",recall_score(y, predy))
            print("Confusion Matrix:",confusion_matrix(y, predy))
            print('\n\n')

        except Exception as E:
            print('Somethings wrong I can feel it : ', E)
        
        iter+=1


iter =1
# for i in range(1,500,50):
#     lm_model = lm.LogisticRegression(warm_start=True,max_iter=i,tol=1e-6)
#     lines.foreachRDD(lambda time, rdd: rdd_test(time, rdd))
#     ssc.start()
#     ssc.awaitTermination()
lm_model=lm.LogisticRegression(warm_start=True,max_iter=50,tol=1e-6)

clf=GridSearchCV(lm.LogisticRegression(),param_grid={'max_iter':[10,50]}, scoring='r2')

lines = lines.flatMap(lambda l: l.split('\\n",'))
lines = lines.map(lambda l:l[2:])
lines = lines.map(lambda l:l.split(',',1))		
lines.foreachRDD(rdd_test)

ssc.start()
ssc.awaitTermination(100000)
ssc.stop()



filename = 'lm_model.sav'
pickle.dump(lm_model, open(filename, 'wb'))
