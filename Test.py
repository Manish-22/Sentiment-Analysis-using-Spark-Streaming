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

sc=SparkContext.getOrCreate()
ssc = StreamingContext(sc, 5)
sqlContext = SQLContext(sc) #required to create dataframe


lines = ssc.socketTextStream("localhost", 6100)

loaded_model = pickle.load(open('./model_multi.pkl', 'rb'))
def rdd_test(time,rdd):

    print(f"===================={str(time)}================")
	
    if rdd.isEmpty():
        pass		
    else:
        try:
            df = sqlContext.createDataFrame(rdd, ['col1','col2'])
            df.dropna()
            tokenizer = Tokenizer(inputCol="col2", outputCol="words")
            hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
            idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
            label_stringIdx = StringIndexer(inputCol = "col1", outputCol = "label")
            pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])
            pipelineFit = pipeline.fit(df)
            train_df = pipelineFit.transform(df)
            train_df.show()
            predictions=loaded_model.score(train_df['features'],train_df['label'])
            print(predictions)

            #print("===================Predictions===================")

            # accuracy = predictions.filter(predictions["label"] == predictions["prediction"]).count() / float(predictions.count())
            # auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
            # print("Accuracy Score: {0:.4f}".format(accuracy))
            # print("ROC-AUC: {0:.4f}".format(auc))

        except Exception as E:
            print('Somethings wrong I can feel it : ', E)


lines = lines.flatMap(lambda l: l.split('\\n",'))
lines = lines.map(lambda l:l[2:])
lines = lines.map(lambda l:l.split(',',1))		# split only by first ,
lines.foreachRDD(rdd_test)


ssc.start()
ssc.awaitTermination(60)
ssc.stop()