import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt

def group_data(df):
    grouped = df.groupby(["Handle", "Party"])["Tweet"].sum()
    grouped = grouped.reset_index()
    return grouped

def naive_bayes(data):
    party = data.loc[:, "Party"]
    tweets = data.loc[:, "Tweet"]
    tweets_train, tweets_test, party_train, party_test = train_test_split(tweets, party, test_size=0.25)
    vectorizer = CountVectorizer()
    classifier = train_bayes(tweets_train, party_train, vectorizer)
    predictions = classifier.predict(vectorizer.transform(tweets_test))
    acc = accuracy_score(party_test, predictions)
    print("Test Accuracy: " + str(acc))
    matrix_display(party_test, predictions)
    classify_public_figures(classifier, vectorizer)

def train_bayes(tweets_train, party_train, vectorizer):
    counts = vectorizer.fit_transform(tweets_train.values)
    classifier = MultinomialNB()
    target = party_train.values
    classifier.fit(counts, target)
    # Writing classifier to a pickle
    save_classifier(classifier)
    return classifier

def save_classifier(classifier):
    """
    Saves classifier into a pickle
    """
    with open('naive_classifier.pickle', 'wb') as f:
        pickle.dump(classifier, f)
    
def classify_public_figures(classifier, vectorizer):
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


def matrix_display(party_test, predictions):
    '''
    This function takes true labels and the  predictions from the data
    used to test the naive bayes classifier, and creates a confusion
    matrix showing the accuracy of the classifier.
    '''
    from sklearn.metrics import confusion_matrix
    party_labels = ['Democrat', 'Republican']
    c_matrix = confusion_matrix(party_test, predictions, party_labels)
    
    # generate the plot
    fig = plt.figure( figsize=(7,7))
    ax = fig.add_subplot(111)
    plt.title('Naive Bayes Model Confusion Matrix')
    ax.set_xticklabels([''] + party_labels) 
    ax.set_yticklabels([''] + party_labels)
    ax.ylabel='True Party'
    ax.xlabel='Models Predicted Party'
    image = ax.matshow(c_matrix, cmap='BuPu')
    plt.xlabel('Predicted Party')
    plt.ylabel('Actual Party')
    
    # place text for the number of correct and incorrect predictions
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            ax.text(j, i, str(c_matrix[i, j]), ha="center", va="center", color="black")
    plt.show()


def main():
    extracted_tweets = pd.read_csv('ExtractedTweets.csv')
    big_data = group_data(extracted_tweets)
    naive_bayes(big_data)

if __name__ == "__main__":
    main()