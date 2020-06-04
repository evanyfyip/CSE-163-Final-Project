import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def group_data(df):
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
        plot_accuracy_bar(party_test, predictions)
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
    Saves classifier and vectorizer into a pickle.
    Parameters:
        classifer: naive bayes classifier object
        vectorizer: Count vectorizer object
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
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.title('Naive Bayes Model Confusion Matrix')
    ax.set_xticklabels([''] + party_labels) 
    ax.set_yticklabels([''] + party_labels)
    ax.ylabel ='True Party'
    ax.xlabel ='Models Predicted Party'
    image = ax.matshow(c_matrix, cmap='BuPu')
    plt.xlabel('Predicted Party')
    plt.ylabel('Actual Party')

    # place text for the number of correct and incorrect predictions
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            ax.text(j, i, str(c_matrix[i, j]), ha="center", va="center", color="black")
    plt.savefig("test_size_" + str(test_size) + ".png")


def plot_accuracy_bar(party_test, predictions):
    """
    Saves and plots the accuracy bar plot of
    the test data and the political affiliation predictions.
    Parameters:
        party_test: a pandas series of the labels of the test data
        predictions: a numpy array of the predictions of the labels
            of the test data
    """
    # Converting party_test and predictions into dataframes
    pred = pd.DataFrame(predictions, columns=['Predictions'])
    test_df = pd.DataFrame(list(party_test), columns=["test_labels"])
    # Merging the two dataframes
    df = test_df.merge(pred, left_index=True, right_index=True)
    # Adding an accuracy column
    df["Accuracy"] = df["test_labels"] == df["Predictions"]

    # Extracting Democrat accuracies
    dem = df['test_labels'] == "Democrat"
    all_dems = df[dem]
    correct_dem = df[dem & df['Accuracy']]

    # Extracting Republican Accuracies
    rep = df['test_labels'] == "Republican"
    all_reps = df[rep]
    correct_rep = df[rep & df['Accuracy']]

    # Determining scalar values of dem_T (prediction = true)
    # and dem_F (prediction = false)
    dem_T = len(correct_dem)
    rep_T = len(correct_rep)
    dem_F = len(all_reps) - dem_T
    rep_F = len(all_dems) - dem_T

    # labels and heights for bar plot
    labels = ['Democrat', 'Republican']
    correct = [dem_T, rep_T]
    incorrect = [dem_F, rep_F]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, correct, width, label='Correct Prediction')
    rects2 = ax.bar(x + width/2, incorrect, width,
                    label='Incorrect Prediction')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Tweets')
    ax.set_xlabel('Political affiliation')
    ax.set_title('Naive Bayes model political affiliation prediction accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    # Generating bar labels
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    # setting layout
    fig.tight_layout()

    plt.show()
    # Saving the figure
    fig.savefig('Political_affiliation_accuracy_bar.png')


def autolabel(rects, ax):
    """
    Attach a text label above each bar in *rects*, displaying its height
    onto the ax object.
    Parameters:
        rects: bar objects
        ax: axis object
    Returns:
        None
    """
    # for each bar in rects
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset from bar
                    textcoords="offset points",
                    ha='center', va='bottom')


def main():
    extracted_tweets = pd.read_csv('ExtractedTweets.csv')
    big_data = group_data(extracted_tweets)
    naive_bayes(big_data)

if __name__ == "__main__":
    main()