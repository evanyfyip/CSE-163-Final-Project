import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def group_data(file_name):
    df = pd.read_csv('ExtractedTweets.csv')
    grouped = df.groupby(["Handle", "Party"])["Tweet"].sum()
    grouped = grouped.reset_index()
    return grouped

def naive_bayes(data):
    party = data.loc[:, "Party"]
    tweets = data.loc[:, "Tweet"]
    accuracy_map = {}
    for i in range(8):
        test_size = round(0.25 + i * 0.1, 2)
        tweets_train, tweets_test, party_train, party_test = train_test_split(tweets, party, test_size=test_size)
        vectorizer = CountVectorizer()
        classifier = train_bayes(tweets_train, party_train, vectorizer)
        predictions = classifier.predict(vectorizer.transform(tweets_test))
        acc = accuracy_score(party_test, predictions)
        accuracy_map[test_size] = acc
        matrix_display(party_test, predictions, test_size)
    accuracy_df = pd.DataFrame(list(accuracy_map.items()), columns = ["Test Size", "Accuracy"])
    print(accuracy_df)
    sns.relplot(x="Test Size", y="Accuracy", data=accuracy_df)
    plt.title("Accuracy vs. Test Size")
    plt.savefig("accuracy_by_test_size.png")

def train_bayes(tweets_train, party_train, vectorizer):
    counts = vectorizer.fit_transform(tweets_train.values)
    classifier = MultinomialNB()
    target = party_train.values
    classifier.fit(counts, target)
    # Writing model to a pickle
    save_model(classifier, vectorizer)
    return classifier

def save_model(classifier, vectorizer):
    """
    Saves classifier into a pickle
    """
    with open('naive_classifier.pickle', 'wb') as f:
        pickle.dump(classifier, f)
    with open('naive_vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)
    
def matrix_display(party_test, predictions, test_size):
    '''
    This function takes true labels and the predictions from the data
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
    plt.savefig("test_size_" + str(test_size) + ".png")


def main():
    big_data = group_data('ExtractedTweets.csv')
    naive_bayes(big_data)

if __name__ == "__main__":
    main()