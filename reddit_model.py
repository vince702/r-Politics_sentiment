
from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import CountVectorizer
import cleantext
from pyspark.sql.types import ArrayType, StringType
# Bunch of imports (may need more)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidatorModel, CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, BooleanType

states  = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
    'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 
    'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 
    'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

def inStates(s):
	return s in states

    
    


#TASK 4 and 5- function to join the grams returned from sanitize() for use in training our model
def sanitize_1(text):
    parsed, unigrams, bigrams, trigrams = cleantext.sanitize(text)
    unigram_array = unigrams.split(" ")
    bigram_array = bigrams.split(" ")
    trigram_array = trigrams.split(" ")

    joined_grams = unigram_array + bigram_array + trigram_array
    return joined_grams



def main(context):
    """Main function takes a Spark SQL context."""
    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED

    #TASK 1
    #https://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html - reading/writing parquet section
    try:
    	#if the parquets have already been generated
    	comments = context.read.parquet("comments.parquet")
    	submissions = context.read.parquet("submissions.parquet")
    	labels = context.read.parquet("labels.parquet")

    except:

    	#if they haven't
        comments = context.read.json("comments-minimal.json.bz2")
        comments.write.parquet("comments.parquet")

        submissions = context.read.json("submissions.json.bz2")
        submissions.write.parquet("submissions.parquet")

        labels = context.read.format('csv').options(header='true', inferSchema='true').load("labeled_data.csv")
        labels.write.parquet("labels.parquet")


    #TASK 2
    #adapted from documentation at https://spark.apache.org/docs/2.2.0/sql-programming-guide.html in the "Running SQL Queries Programmatically" section
    labels.createOrReplaceTempView("labels")
    comments.createOrReplaceTempView("comments")

    labeled_comments = context.sql("SELECT labels.Input_id, body, labels.labeldem, labels.labelgop, labels.labeldjt FROM comments JOIN labels ON id=Input_id ")
    #labeled_comments = context.sql("SELECT labels.Input_id, body, labels.labeldem, labels.labelgop, labels.labeldjt FROM comments JOIN labels ON id=Input_id LIMIT 5")
    #labeled_comments.show()
    labeled_comments.createOrReplaceTempView("labeled_comments")


    #TASK 4 and 5
    #source on syntax for how to write a UDF:  https://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html
    context.registerFunction("sanitize", sanitize_1, ArrayType(StringType()))

    training_data = context.sql("SELECT Input_id, sanitize(body) AS body, labeldem, labelgop, labeldjt FROM labeled_comments")
    #training_data.show()

    #TASK 6a and 6b

    #source on countvectorizer code: https://spark.apache.org/docs/latest/ml-features.html- in the CountVectorizer section

    cv = CountVectorizer(inputCol="body", outputCol="features", minDF=10.0, binary=True)
    model = cv.fit(training_data)
    training_data_label = model.transform(training_data)
    #result.show(truncate=False)
    training_data_label.createOrReplaceTempView("training_data_label")

    #only using DJT 
    pos = context.sql("SELECT *, if(labeldjt = 1,1,0) AS label FROM training_data_label ")
    neg = context.sql("SELECT *, if(labeldjt = -1,1,0) AS label FROM training_data_label ")

    pos.show()
    neg.show()
    

    #try:

    #Task 7



    try:
        posModel = CrossValidatorModel.load("project2/pos.model")
        negModel = CrossValidatorModel.load("project2/neg.model")

    except:

        #code from spec

        # Initialize two logistic regression models.
        # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
        poslr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
        neglr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
        # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
        posEvaluator = BinaryClassificationEvaluator()
        negEvaluator = BinaryClassificationEvaluator()
        # There are a few parameters associated with logistic regression. We do not know what they are a priori.
        # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
        # We will assume the parameter is 1.0. Grid search takes forever.
        posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
        negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
        # We initialize a 5 fold cross-validation pipeline.
        posCrossval = CrossValidator(
            estimator=poslr,
            evaluator=posEvaluator,
            estimatorParamMaps=posParamGrid,
            numFolds=5)
        negCrossval = CrossValidator(
            estimator=neglr,
            evaluator=negEvaluator,
            estimatorParamMaps=negParamGrid,
            numFolds=5)
        # Although crossvalidation creates its own train/test sets for
        # tuning, we still need a labeled test set, because it is not
        # accessible from the crossvalidator (argh!)
        # Split the data 50/50
        posTrain, posTest = pos.randomSplit([0.5, 0.5])
        negTrain, negTest = neg.randomSplit([0.5, 0.5])
        # Train the models
        print("Training positive classifier...")
        posModel = posCrossval.fit(posTrain)
        print("Training negative classifier...")
        negModel = negCrossval.fit(negTrain)
    
        # Once we train the models, we don't want to do it again. We can save the models and load them again later.
        posModel.save("project2/pos.model")
        negModel.save("project2/neg.model")





    comments = context.read.parquet("comments.parquet")
    comments.createOrReplaceTempView("comments")
    submissions.createOrReplaceTempView("submissions")


    #TASK 8 and 9
    comments_filtered = context.sql("SELECT * FROM comments WHERE comments.body NOT LIKE '%/s%' and comments.body NOT LIKE '&gt%'")
    comments_filtered.createOrReplaceTempView("comments_filtered")

    unseen_df = context.sql("SELECT comments_filtered.link_id AS id, comments_filtered.body, comments_filtered.created_utc, "+
    	  "comments_filtered.author_flair_text, submissions.title, submissions.score AS submission_score, comments_filtered.score AS comments_score FROM"+
    	  " comments_filtered JOIN submissions ON replace(comments_filtered.link_id, 't3_','')=submissions.id")
    

    unseen_df.createOrReplaceTempView("unseen_df")

    context.registerFunction("sanitize", sanitize_1, ArrayType(StringType()))


    unseen_data = context.sql("SELECT id, created_utc, author_flair_text, title, submission_score,"+
    	" comments_score, sanitize(body) AS body FROM unseen_df")
    
    unseen = model.transform(unseen_data)


    #source for classification syntax on a dataframe: https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html 

    # source for accessing element of probability array:
    #https://stackoverflow.com/questions/44425159/access-element-of-a-vector-in-a-spark-dataframe-logistic-regression-probability
    second=udf(lambda v:float(v[1]),FloatType())

    positivePredictions = posModel.transform(unseen)





    #used method provided in tips on PIAZZA instead of a join 

    positivePredictions = positivePredictions.select("features", "id", "created_utc", "title", "author_flair_text", "body", "submission_score", 
    	                            "comments_score", second('probability').alias('pos_prob'))

    negativeAndPositivePredictions = negModel.transform(positivePredictions)



    negativeAndPositivePredictions = negativeAndPositivePredictions.select("features", "id", "created_utc", "title", "author_flair_text", "body", 
                                     "submission_score", "comments_score", "pos_prob", second('probability').alias('neg_prob'))
    
    negativeAndPositivePredictions.createOrReplaceTempView("negativeAndPositivePredictions")

    predictions = context.sql("SELECT id, title, created_utc, author_flair_text, submission_score, comments_score, "+
    	                    "if(pos_prob > .2,1,0) as pos_label, if(neg_prob > .25,1,0) as neg_label FROM negativeAndPositivePredictions")

   


    
    try:
        predictions = context.read.parquet("predictions.parquet")
    except:
    	predictions.write.parquet("predictions.parquet")
    	predictions = context.read.parquet("predictions.parquet")

    predictions.createOrReplaceTempView("predictions")
    predictions.show()
    #save predictions as parquet -- without this writing the task 10 dfs to disk takes around 16 hours, with it, including writing predictions,
    #takes around 2.5 hrs total



    #TASK 10

    submission_percentage = context.sql("SELECT id, AVG(pos_label) as pos, AVG(neg_label) as neg FROM predictions GROUP BY id")

    submission_percentage.createOrReplaceTempView("submission_percentage")
    top_pos_submission =  context.sql("SELECT id, pos, neg FROM submission_percentage ORDER BY pos DESC, neg ASC LIMIT 10")
    top_neg_submission =  context.sql("SELECT id, pos, neg FROM submission_percentage ORDER BY neg DESC, pos ASC LIMIT 10")

    top_pos_submission.createOrReplaceTempView("top_pos_submission")
    top_neg_submission.createOrReplaceTempView("top_neg_submission")


  

    time_data = context.sql("SELECT from_unixtime(created_utc, 'yyyy-MM-dd') AS date, AVG(pos_label) AS Positive, AVG(neg_label) AS Negative"+
    	                    " FROM predictions GROUP BY date")
   
    
    context.registerFunction("inStates", inStates, BooleanType())

    state_data = context.sql("SELECT author_flair_text AS state, AVG(pos_label) AS Positive, AVG(neg_label) AS Negative"+
    	                        " FROM predictions WHERE(inStates(author_flair_text)) GROUP BY author_flair_text")

    comment_score = context.sql("SELECT comments_score as comment_score, AVG(pos_label) AS Positive, AVG(neg_label) AS Negative"+
    	                        " FROM predictions GROUP BY comments_score")
 

    submission_score = context.sql("SELECT submission_score, AVG(pos_label) AS Positive, AVG(neg_label) AS Negative"+
    	                           " FROM predictions GROUP BY submission_score")


    
    #num of posts by state
    state_count_data = context.sql("SELECT author_flair_text AS state, COUNT(pos_label) "+
    	                        " FROM predictions WHERE(inStates(author_flair_text)) GROUP BY author_flair_text")


    



    time_data.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("time_data.csv")
    state_data.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("state_data.csv")
    comment_score.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("comment_score.csv")
    submission_score.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("submission_score.csv")
    state_count_data.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("state_count_data.csv")


    top_pos_submission =  context.sql("SELECT submissions.title, top_pos_submission.id, top_pos_submission.pos, top_pos_submission.neg FROM top_pos_submission JOIN submissions on replace(top_pos_submission.id, 't3_','')= submissions.id")
    top_neg_submission =  context.sql("SELECT submissions.title, top_neg_submission.id, top_neg_submission.pos, top_neg_submission.neg FROM top_neg_submission JOIN submissions on replace(top_neg_submission.id, 't3_','')= submissions.id") 
    
  

    
    top_pos_submission.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("top_pos_submission.csv")
    top_neg_submission.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("top_neg_submission.csv")

    context.sql("SELECT submissions.title, top_neg_submission.id, top_neg_submission.pos, top_neg_submission.neg FROM top_neg_submission JOIN submissions on replace(top_neg_submission.id, 't3_','')= submissions.id").explain()
    




if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)