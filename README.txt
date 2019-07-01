Vincent Chi
304576879

Late days used: 1

Extra plots:
Number of posts to r/pol by state to see why it's so negative towards trump, and results:
Ton of posts from California especially and some other left leaning states

Notes:
*Using the Code provided in the spec when saving to disk into CSV files, the file names are used as the folder names instead, meaning time_data.csv is the name of the folder, which contains a. File called SUCCESS and a funky-named csv file with the actual data.

*Assumes clean workspace, code takes around 1 hr to train the model on my computer, 1 hr to load the 3 files given, and another 2 hrs to write the predicted values of all the unseen comments to disk as a parquet, and then around 30 minutes to generate the csv files afterwards.
(quad core, 16gb ram although not sure how much is given to the VM)


*All Sources including from the official documentation are cited within code, the only source that wasn’t official documentation was 
    #https://stackoverflow.com/questions/44425159/access-element-of-a-vector-in-a-spark-dataframe-logistic-regression-probability
- the UDF to access index 1 or probability
Which was used after running into a roadblock when trying to access index 1 of the probability array returned by the logistic regression model


