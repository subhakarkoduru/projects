
# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import Rating
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import split, collect_list
from pyspark.sql.functions import explode, col
import matplotlib.pyplot as plt
from pyspark.sql.functions import desc,lit,udf,regexp_replace
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import IDF, Tokenizer
from pyspark.ml.feature import CountVectorizer, Tokenizer, IDF
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import explode
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType
import random
from collections import Counter
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import IndexToString


spark = SparkSession.builder.appName("Tags prediction")\
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "12g") \
    .getOrCreate()

# Loading the 'Tags.csv' file
tags = spark.read.csv("Tags.csv", header=True,sep=",",inferSchema=True)
tags.show()

# Grouping 'Tags' by 'Id'
grouped_df = tags.groupBy("Id").agg(collect_list("Tag").alias("combined_tags"))
grouped_df.show(truncate=False)

# Loading the 'Questions.csv' file
file_path = "Questions.csv"
df = spark.read.options(header=True, inferSchema=True, quotes='"', escape="\"", multiLine=True).csv(file_path)
df= df.drop(df.ClosedDate,df.CreationDate,df.OwnerUserId)

Question_tag = df.join(grouped_df, on="Id", how="inner")
Question_tag = Question_tag.filter((col("Score") > 5))

# Filtering duplicate 'Id's 
duplicate_count = Question_tag.groupBy(Question_tag.Id).count().filter("count > 1")
duplicate_count.show()

# Extracting distinct tags and their counts and collect best tags
all_tags_df = Question_tag.select(explode("combined_tags").alias("tag"))
all_tags_df_distint = all_tags_df.distinct()
all_tags_df_with_count = all_tags_df.withColumn("count", lit(1))
tag_counts = all_tags_df_with_count.groupBy("tag").count()

# # Group by tag and count the occurrences
# tag_counts = all_tags_df_with_count.groupBy("tag").count()

# # Sort by count in descending order and select the top 100
# top_100_tags = tag_counts.orderBy(desc("count")).limit(1000)


# # Convert PySpark DataFrame to Pandas DataFrame for plotting
# pandas_top_100_tags = top_100_tags.toPandas()

# # Plot the line graph for the top 100 tags
# plt.figure(figsize=(12, 6))
# plt.plot(pandas_top_100_tags["tag"], pandas_top_100_tags["count"], marker='o', linestyle='-', color='b')
# plt.xlabel("Tag")
# plt.ylabel("Count")
# plt.title("Top 1000 Tags by Count (Line Graph)")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt
# import pandas as pd

# # Convert the Spark DataFrame to a Pandas DataFrame
# tag_counts_pd = tag_counts.toPandas()

# # Plot the histogram with adjustments
# plt.figure(figsize=(10, 6))
# # Use log scale if needed by setting log=True
# plt.hist(tag_counts_pd['count'], bins=50, color='blue', alpha=0.7, log=True)
# plt.title('Distribution of Tag Counts')
# plt.xlabel('Tag Count')
# plt.ylabel('Number of Tags (log scale)')
# plt.grid(True)
# plt.show()

threshold = 800  
filtered_tags = tag_counts.filter(col("count") >= threshold)
sorted_tags = filtered_tags.orderBy(col("count").desc())
best_tags_list = sorted_tags.collect()
exploded_df = Question_tag.select("Id","Title", "Body", explode("combined_tags").alias("tag"))
filtered_df = exploded_df.join(sorted_tags, "tag", "inner").groupBy("Id", "Title", "Body") \
    .agg(collect_list("tag").alias("filtered_tags"))
filtered_df.show(truncate=False)

#remove html tags
def remove_html_tags(text):
    clean_text = F.regexp_replace(text, "<.*?>", "")
    return clean_text
filtered_df = filtered_df.withColumn("HtmlR_body", remove_html_tags(col("Body")))
filtered_df.show()
#remove puntuation
filtered_df = filtered_df.withColumn("body_without_punctuation", regexp_replace(filtered_df["HtmlR_body"], r'[^\w\s]', ''))
filtered_df = filtered_df.withColumn("Title_without_punctuation", regexp_replace(filtered_df["Title"], r'[^\w\s]', ''))
#tokenize data
tokenizer = Tokenizer(inputCol="body_without_punctuation", outputCol="body_tokens")
tokenizer1 = Tokenizer(inputCol="Title_without_punctuation", outputCol="title_tokens")
filtered_df = tokenizer.transform(filtered_df)
filtered_df = tokenizer1.transform(filtered_df)
#remove stopwords
stopwords_remover = StopWordsRemover(inputCol="body_tokens", outputCol="filtered_body_tokens")
filtered_df = stopwords_remover.transform(filtered_df)
stopwords_remover = StopWordsRemover(inputCol="title_tokens", outputCol="filtered_title_tokens")
filtered_df = stopwords_remover.transform(filtered_df)
#explode and collect distinct tags
exploded_df = filtered_df.select("Id", "Title", explode("filtered_tags").alias("tag"))
exploded_df.show()
distinct_tags = exploded_df.select("tag").distinct()
#convert unique tags into indexed labels 
string_indexer = StringIndexer(inputCol="tag", outputCol="indexed_label")
indexer_model = string_indexer.fit(distinct_tags)
indexed_tags = indexer_model.transform(distinct_tags) 
indexed_tags.show()
exploded_df = exploded_df.join(indexed_tags, on="tag", how="left")
exploded_df.show()
filtered_df = exploded_df.join(filtered_df, on="Id", how="left")
#drop duplicate ids
filtered_df = filtered_df.dropDuplicates(['Id'])
filtered_df.show()
#perform TF-IDF on title tokens
vectorizer_title = CountVectorizer(inputCol="filtered_title_tokens", outputCol="title_rawFeatures", minDF=3, vocabSize=5000)
model_title = vectorizer_title.fit(filtered_df)
featurizedData_title = model_title.transform(filtered_df)

idf_title = IDF(inputCol="title_rawFeatures", outputCol="title_features")
idfModel_title = idf_title.fit(featurizedData_title)
rescaledData = idfModel_title.transform(featurizedData_title)
rescaledData.show(5)

#perform TF-IDF on body tokens
vectorizer_body = CountVectorizer(inputCol="filtered_body_tokens", outputCol="body_rawFeatures", minDF=3, vocabSize=5000)
model_body = vectorizer_body.fit(filtered_df)
featurizedData_body = model_body.transform(filtered_df)

idf_body = IDF(inputCol="body_rawFeatures", outputCol="body_features")
idfModel_body = idf_body.fit(featurizedData_body)
rescaledData_body = idfModel_body.transform(featurizedData_body)
rescaledData_body.show(5)
rescaledData = rescaledData.select("Id", "title_features","indexed_label").join(rescaledData_body.select("Id", "body_features"), on='Id', how='inner')
rescaledData.show()

#assemble features
assembler = VectorAssembler(
    inputCols=["title_features", "body_features"],
    outputCol="features"
)

rescaledData = assembler.transform(rescaledData)
#perform LDA
lda = LDA(k=10, maxIter=10)
ldaModel = lda.fit(rescaledData)

transformed_data = ldaModel.transform(rescaledData)
transformed_data.show()

#combine LDA with previous feaures
assembler1 = VectorAssembler(
    inputCols=["features", "topicDistribution"],
    outputCol="topic_features"
)

transformed_data = assembler1.transform(transformed_data)
#split the dataset
train_data, test_data = transformed_data.randomSplit([0.8, 0.2], seed=42)

#build random forest using pipeline
rf = RandomForestClassifier(labelCol="indexed_label", featuresCol="topic_features")

pipeline = Pipeline(stages=[rf])
paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [12, 15, 20])  
             .addGrid(rf.maxDepth, [30, 12, 40])  
             .build())
evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="f1")
recallevaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="recallByLabel")
#perform crossvalidation
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)  
cvModel = crossval.fit(train_data)

predictions = cvModel.transform(test_data)
predictions.show(70)

# evaluate f1 score and recall
f1_score = evaluator.evaluate(predictions)
recall = recallevaluator.evaluate(predictions)
print(f"F1 Score: {f1_score}")

print(f"recall Score: {recall}")

evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction")

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print("Accuracy:", accuracy)
print("Weighted Precision:", precision)
print("Weighted Recall:", recall)
print("F1 Score:", f1)

#impleent MLP using different layers 
layers = [len(transformed_data.select('topic_features').first()[0]),128, 64, len(rescaledData.select('indexed_label').distinct().collect())]

mlp = MultilayerPerceptronClassifier(layers=layers, labelCol='indexed_label', featuresCol='topic_features', seed=42)

mlp_model = mlp.fit(train_data)

predictions1 = mlp_model.transform(test_data)

predictions1.show()
evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", metricName="f1")
recallevaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="recallByLabel")
# evaluate f1 score and recall
accuracy = evaluator.evaluate(predictions1)
print("f1 score:", accuracy)
recall = recallevaluator.evaluate(predictions)
print(f"recall Score: {recall}")

evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction")

accuracy = evaluator.evaluate(predictions1, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions1, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions1, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions1, {evaluator.metricName: "f1"})

print("Accuracy:", accuracy)
print("Weighted Precision:", precision)
print("Weighted Recall:", recall)
print("F1 Score:", f1)

#implement logistic regression
lr = LogisticRegression(
    labelCol="indexed_label", featuresCol="topic_features",regParam=0.1
)

lrModel = lr.fit(train_data)

predictions2 = lrModel.transform(test_data)
predictions2.show(70)
evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="f1")
recallevaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="recallByLabel")

# evaluate f1 score and recall
recall = recallevaluator.evaluate(predictions)
print(f"recall Score: {recall}")
lr_accuracy = evaluator.evaluate(predictions2)
print(f"F1 Score: {lr_accuracy}")

evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction")

accuracy = evaluator.evaluate(predictions2, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions2, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions2, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions2, {evaluator.metricName: "f1"})

print("Accuracy:", accuracy)
print("Weighted Precision:", precision)
print("Weighted Recall:", recall)
print("F1 Score:", f1)

joined_df = predictions.alias("pred").join(
    predictions1.alias("pred1"),
    on=["Id"],
    how="inner"
).join(
    predictions2.alias("pred2"),
    on=["Id"],
    how="inner"
)

#voting method for hybrid model
def vote(column1, column2, column3):
    votes = Counter([column1, column2, column3])
    winner, _ = votes.most_common(1)[0]
    return winner

vote_udf = udf(vote, StringType())

#hybrid model using previous 3 models
hybrid_predictions = joined_df.withColumn(
    "supervised_hybrid_prediction",
    vote_udf(
        col("pred.prediction"),
        col("pred1.prediction"),
        col("pred2.prediction")
    )
)
hybrid_predictions.show()

hybrid_predictions = hybrid_predictions.withColumn("supervised_hybrid_prediction", col("supervised_hybrid_prediction").cast(DoubleType()))

predictionAndLabels = hybrid_predictions.select(['supervised_hybrid_prediction', 'pred1.indexed_label'])

# evaluate f1 score and recall
evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="supervised_hybrid_prediction")
recall = evaluator.evaluate(predictionAndLabels, {evaluator.metricName: "recallByLabel"})
f1 = evaluator.evaluate(predictionAndLabels, {evaluator.metricName: "f1"})
precision = evaluator.evaluate(predictionAndLabels, {evaluator.metricName: "weightedPrecision"})
print("Recall = %s" % recall)
print("F1 Score = %s" % f1)
print("precision= %s" % precision)
#convert indexed labels back to string original tags 
labelConverter = IndexToString(inputCol="supervised_hybrid_prediction", outputCol="tag_pred", labels=indexer_model.labels)
hybrid_predictions = labelConverter.transform(hybrid_predictions)

hybrid_predictions.show()