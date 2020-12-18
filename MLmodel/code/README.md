# Overview of Machine Learning related Code
* datapreparation.py: construct labels from the labeled script data
* run\_models.py: grid search functions for all machine learning models with evaluation
* run\_round0.py: the initial round to call functions in run\_models.py and compare the predictions with the ground truth to find out inconsistency
* run\_round1.py: another round to call functions after the corrections by the initial round and report the performance
* important\_feat\_analysis.py: statistics of significant feautres, such as the ratio of occurances in both positive and negative classes
