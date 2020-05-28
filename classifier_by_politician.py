import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle

def group_data(df):
    grouped = df.groupby(["Handle", "Party"])["Tweet"].sum()
    grouped = grouped.reset_index()
    return grouped

def main():
    extracted_tweets = pd.read_csv('ExtractedTweets.csv')
    big_data = group_data(extracted_tweets)
    with open('scraped_tweets.pickle', 'rb') as f:
        scraped_tweets = pickle.load(f)
    pub_figures = scraped_tweets.groupby(["username"])["tweet"].sum().reset_index()

    party = big_data.iloc[:, 1]
    tweets = big_data.iloc[:, 2]

    public_figures = pub_figures.iloc[:, 0]
    test_tweets = pub_figures.iloc[:, 1]

    vectorizer = CountVectorizer()
    tweets_train, tweets_test, party_train, party_test = train_test_split(tweets, party, test_size=0.25)
    counts = vectorizer.fit_transform(tweets_train.values)
    classifier = MultinomialNB()
    target = party_train.values
    classifier.fit(counts, target)
    prediction = classifier.predict(vectorizer.transform(tweets_test))
    prediction2 = classifier.predict(vectorizer.transform(test_tweets))
    acc = accuracy_score(party_test, prediction)
    print("Test Accuracy: " + str(acc))

    map = {}
    for i in range(len(public_figures)):
        map[public_figures[i]] = prediction2[i]
    print(map)

if __name__ == "__main__":
    main()