CSE-163-Final-Project

Predicting Political Affiliation using tweets
Group Members:
Evan Yip
Walker Azam
Cooper Chia

Files: ExtractedTweets.csv, naive_bayes_classifier.py,
classify_public_figures.py, scraped_tweets.pickle

ExtractedTweets.csv:
CSV file containing extracted tweets from select politicians.
It has three columns: handle, party, and tweet. The dataset was
taken from Kaggle. It is imported in naive_bayes_classifier.py
so it must be located in the same directory.

naive_bayes_classifier.py:
This file contains functions to train and test multiple
multinomial Naive Bayes classifier models (based on different
testing-training split). It must be in the same directory as
ExtractedTweets.csv, since it imports it as a dataframe.
To run this function you just need to press the run button or
type 'python naive_bayes_classifier.py' in the terminal.
It will save multiple plots into the same directory as the
file, which visualise the performance of the different models.
