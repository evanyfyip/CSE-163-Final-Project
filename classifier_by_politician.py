import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

def main():
    df = pd.read_csv('ExtractedTweets.csv')
    grouped = df.groupby(["Handle", "Party"])["Tweet"].sum()
    grouped = grouped.reset_index()

    party = grouped.iloc[:, 1]
    tweets = grouped.iloc[:, 2]

    tweets_train, tweets_test, party_train, party_test = train_test_split(tweets, party, test_size=0.25)
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(tweets_train.values)

    classifier = MultinomialNB()
    target = party_train.values

    classifier.fit(counts, target)
    prediction = classifier.predict(vectorizer.transform(tweets_test))
    acc = accuracy_score(party_test, prediction)

    print(acc)

if __name__ == "__main__":
    main()