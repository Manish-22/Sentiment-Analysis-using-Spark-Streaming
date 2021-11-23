import findspark
#findspark.init()
import time
from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row,SQLContext
import sys
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split

spark=SparkSession.builder.appName("bIG DATA").getOrCreate()

#Uncomment this for datastreaming
"""
# conf=SparkConf()
# conf.setAppName("BigData")
# sc=SparkContext.getOrCreate(conf=conf)

# ssc=StreamingContext(sc,5)

# dataStream=ssc.socketTextStream("localhost",6100)
# dataStream.
# ssc.start()
# ssc.awaitTermination()
# ssc.stop()
"""

# lines = spark.readStream.format("socket").option("host", "localhost").option("port", 6100).load()
# words = lines.select(
#    explode(
#        split(lines.value, " ")
#    ).alias("word")
# )
# lines.

# wordCounts = words.groupBy("word")

# query = lines.writeStream.outputMode("append").format("console").option("truncate","false").start()
# di = lines.writeStream()
# query.awaitTermination()

#df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("multiline",True).load("./spam/train.csv").replace(r'\\r','')
df = spark.read.csv('./spam/test.csv', sep=',', escape='"', header=True, 
               inferSchema=True, multiLine=True).withColumnRenamed("Spam/Ham", "label_string").withColumnRenamed("Message", "sms")
df = df.select("label_string","sms").fillna(value='spam')
df1 = df.select("label_string","sms").toPandas()
df.show(20)
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes

stages = []
# 1. clean data and tokenize sentences using RegexTokenizer
regexTokenizer = RegexTokenizer(inputCol="sms", outputCol="tokens", pattern="\\W+")
stages += [regexTokenizer]

# 2. CountVectorize the data
cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)#, vocabSize=3, minDF=2.0
stages += [cv]

# 3. Convert the labels to numerical values using binariser
indexer = StringIndexer(inputCol="label_string", outputCol="label")
stages += [indexer]

# 4. Vectorise features using vectorassembler
vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")
stages += [vecAssembler]

[print('\n', stage) for stage in stages]


from pyspark.ml import Pipeline
pipeline = Pipeline(stages=stages)
data = pipeline.fit(df).transform(df)

train, test = data.randomSplit([0.7, 0.3], seed = 2018)

from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(train)

predictions = model.transform(test)
# Select results to view
predictions.limit(10).select("label", "prediction", "probability").show(truncate=False)


from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print ("Test Area Under ROC: ", accuracy)




from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid and Evaluator for Cross Validation
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]).build()
cvEvaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
# Run Cross-validation
cv = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid, evaluator=cvEvaluator)
cvModel = cv.fit(train)
# Make predictions on testData. cvModel uses the bestModel.
cvPredictions = cvModel.transform(test)
# Evaluate bestModel found from Cross Validation
evaluator.evaluate(cvPredictions)



# Make predictions on testData. cvModel uses the bestModel.
cvPredictions = cvModel.transform(test)
# Evaluate bestModel found from Cross Validation
print ("Test Area Under ROC: ", evaluator.evaluate(cvPredictions))

