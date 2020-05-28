import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    df = pd.read_csv('ExtractedTweets.csv')
    grouped = df.groupby(["Handle", "Party"])["Tweet"].sum()
    df = grouped.reset_index()

    labels = df.iloc[:, 1]
    categories = df.iloc[:, 2]

    category_train, category_test, label_train, label_test = train_test_split(categories, labels, test_size=0.3)
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(category_train.values)

    classifier = MultinomialNB()
    target = label_train.values

    classifier.fit(counts, target)
    prediction = classifier.predict(vectorizer.transform(category_test))
    acc = accuracy_score(label_test, prediction)

    print(len(prediction))
    print(acc)

if __name__ == "__main__":
    main()