"""
Evan Yip, Cooper Chia, Walker Azam
CSE 163 Final Project

This file contains functions to runs Naive Bayes model
on web scraped tweets data to predict their political
affiliation. The model used is from 0.25 test-train split
6/3/2020
"""
import pickle


def classify_public_figures():
    """
    Tests the classifier and vectorizer Naive Bayes Model
    on the web scraped tweets of public figures, utilizing
    the stored pickle files. Prints out the political
    affilitation predictions
    """
    with open('naive_vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)
    # Loading the classifier (0.25 test-train split)
    with open('naive_classifier.pickle', 'rb') as f:
        classifier = pickle.load(f)
    # Loading our web scraped tweets
    with open('scraped_tweets.pickle', 'rb') as f:
        scraped_tweets = pickle.load(f)
    # Grouping tweet data by username and aggregating the sum
    pub_figures = scraped_tweets.groupby(["username"])["tweet"].sum()
    pub_figures = pub_figures.reset_index()
    # Retrieving public figures usernames
    public_figures = pub_figures.iloc[:, 0]
    # retrieving tweets
    test_tweets = pub_figures.iloc[:, 1]
    # Vectorizing test tweets
    vector = vectorizer.transform(test_tweets)
    # Predicting features
    predictions = classifier.predict(vector)
    print(classifier.predict_proba(vector))
    # initializing map dictionary
    map = {}
    for i in range(len(public_figures)):
        map[public_figures[i]] = predictions[i]
    # printing prediction output
    print(map)


def main():
    classify_public_figures()


if __name__ == "__main__":
    main()
