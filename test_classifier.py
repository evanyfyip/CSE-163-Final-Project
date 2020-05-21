import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
df = pd.read_csv('ExtractedTweets.csv', header=0)
df.head(12)

labels = df.iloc[:, 0]
categories = df.iloc[:, 2]

category_train, category_test, label_train, label_test = train_test_split(categories, labels, test_size=0.3)
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(category_train.values)

classifier = MultinomialNB()
target = label_train.values

classifier.fit(counts, target)
prediction = classifier.predict(vectorizer.transform(category_test))
acc = accuracy_score(label_test, prediction)

print(acc)