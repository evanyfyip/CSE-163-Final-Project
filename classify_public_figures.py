import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


def classify_public_figures():
    with open('naive_vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)
    # Loading the classifier
    with open('naive_classifier.pickle', 'rb') as f:
        classifier = pickle.load(f)
    # Loading our web scraped tweets
    with open('scraped_tweets.pickle', 'rb') as f:
        scraped_tweets = pickle.load(f)
    pub_figures = scraped_tweets.groupby(["username"])["tweet"].sum().reset_index()
    public_figures = pub_figures.iloc[:, 0]
    test_tweets = pub_figures.iloc[:, 1]
    predictions = classifier.predict(vectorizer.transform(test_tweets))
    map = {}
    for i in range(len(public_figures)):
        map[public_figures[i]] = predictions[i]
    print(map)

def main():
    classify_public_figures()

if __name__ == "__main__":
    main()