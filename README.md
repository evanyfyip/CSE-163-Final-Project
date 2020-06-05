CSE-163-Final-Project

Predicting Political Affiliation using tweets
Group Members:
Evan Yip
Walker Azam
Cooper Chia

Files: ExtractedTweets.csv, naive_bayes_classifier.py,
classify_public_figures.py, scraped_tweets.pickle,
tweet_scraping_legacy.py

**NOTE**
This specification assumes the user has the standard cse163 environment.

ExtractedTweets.csv:
CSV file containing extracted tweets from select politicians.
It has three columns: handle, party, and tweet. The dataset was
taken from Kaggle (https://www.kaggle.com/kapastor/democratvsrepublicantweets/version/3),
where it can be easily downloaded.
It is imported in naive_bayes_classifier.py
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
It also saves the classifier constructed from a 0.25 test-train
split, as a pickle. This classifier is used in classify_public_figures.py

tweet_scraping_legacy.py:
**NOTE**
tweet_scraping_legacy.py is no longer functional as of June 1st, 2020.
There was an update to Twitter effective as of the date above that
permanently disabled accessing the "legacy" version of twitter.
This has made scraping tweets from twitter a significantly more challenging
task for all future attempts. The following describes the previously functional code:

This file contains functions that scrape tweets from twitter.com from
various public figures and save these tweets and the usernames of the tweeters
to a Pandas dataframe in scraped_tweets.pickle. This pickle will be in the
same directory as tweet_scraping_legacy.py. In order to run this file,
the BeautifulSoup library must be installed, which is not included in the
cse163 environment by default. To install, type "pip install bs4" at the terminal.

scraped_tweets.pickle:
This pickle contains scraped tweets from twitter users and is
used in the python file classify_public_figures.py.

naive_classifier.pickle:
This pickle contains the Naive Bayes Classifier we produced in naive_bayes_classifier.py,
taken from the model of test size 0.25. The model saved in this pickle is used in
classify_public_figures.py to predict the politicial sentiment of public figures.

naive_vectorizer.pickle:
This pickle contains the vectorizer object we used to fit our string data of tweets
when training the classifier. We save it into a pickle that is later read and used
when classifying public figures. This is important to help our model maintain consistency
and function. The pickle should be saved to the svae directory as
classify_public_figures.py

classify_public_figures.py:
This file runs the previously constructed multinomial Naive Bayes
model on the scraped tweets from scraped_tweets.pickle, and
classifies the twitter users. To run it, the pickle must be in the
same directory as classify_public_figures.py. Run the file by pressing
the Run button or typing 'python classify_public_figures.py' in 
the terminal
