"""
Evan Yip, Cooper Chia, Walker Azam
CSE 163 Final Project (June/2020)

This file contains methods to generate a Naive Bayes
classifier to classify twitter users as Democrats
or Republicans based off their tweets
"""
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns


def group_data(df):
    '''
    This helper function groups together tweets from the same person
    and by party. It takes in a dataframe of tweets as a parameter
    and returns a grouped dataframe.
    '''
    grouped = df.groupby(["Handle", "Party"])["Tweet"].sum()
    grouped = grouped.reset_index()
    return grouped


def naive_bayes(data):
    '''
    This function takes a dataframe containing tweets and party affiliation
    from an imported file. It generates 8 different naive bayes models
    according to different test sizes and generates multiple plots to
    visualise their effectiveness in prediction. These plots are saved to
    confusion_matrix.png and accuracy_bar.png. It also saves a plot
    comparing all model's accuracy scores in a plot called
    accuracy_by_test_size.png.
    Parameters:
        data: a pandas data frame containing tweets and party affiliation
            columns
    Returns:
        None
    '''
    # seperating labels and tweets
    party = data.loc[:, "Party"]
    tweets = data.loc[:, "Tweet"]
    accuracy_map = {}  # a dictionary to store the accuracy scores of the model

    # loops through different test sample sizes to test against a trained model
    con_fig, axs1 = plt.subplots(2, 4, figsize=(10, 8))  # confusion matrix fig
    bar_fig, axs2 = plt.subplots(2, 4, figsize=(15, 12))  # bar plot figure
    for i in range(8):
        test_size = round(0.25 + i * 0.1, 2)
        # separating testing and training data accorind to test size
        tweets_train, tweets_test, party_train,\
            party_test = train_test_split(tweets, party, test_size=test_size)
        # generating a vectorizer to handle string inputs
        vectorizer = CountVectorizer()
        # calling function to train model
        classifier = train_bayes(tweets_train, party_train, vectorizer, test_size)
        predictions = classifier.predict(vectorizer.transform(tweets_test))
        acc = accuracy_score(party_test, predictions)
        accuracy_map[test_size] = acc  # storing the accuracy score
        # calling functions to visualise each model according to testing size
        # generating subplot indices
        if i < 4:
            row = 0
            col = i
        else:
            row = 1
            col = i - 4
        # Plotting subplots of confusion matrix and accuracy bar
        matrix_display(party_test, predictions, test_size, axs1[row, col])
        plot_accuracy_bar(party_test, predictions, test_size, axs2[row, col])
    # Plotting and saving confusion matrix
    con_fig.suptitle('NB Model Confusion Matrices with varying test sizes',
                     fontsize=20)
    con_fig.tight_layout()  # tight layout
    con_fig.savefig("confusion_matrix.png")
    # Plotting and saving bar plot
    bar_fig.suptitle('NB Model Accuracy Bar plots with varying test sizes',
                     fontsize=20)
    bar_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    bar_fig.savefig("accuracy_bar.png")

    # creating a new dataframe that stores accuracy scores
    accuracy_df = pd.DataFrame(list(accuracy_map.items()),
                               columns=["Test Size", "Accuracy"])
    print(accuracy_df)
    # plotting accuracy scores of models by their test size
    sns.relplot(x="Test Size", y="Accuracy", data=accuracy_df)
    plt.title("Accuracy vs. Test Size")
    plt.savefig("accuracy_by_test_size.png")


def train_bayes(tweets_train, party_train, vectorizer, test_size):
    '''
    This function trains a multinomial naive bayes classifier model.
    It takes training data and training labels, and the vectorizer
    to handle strings and the test_size, as parameters. It returns a trained
    classifier based on the training data/labels given.
    Parameters:
        tweets_train: the features -- tweets training set
        party_train: the labels of the training set
        vectorizer: countvectorizer object
        test_size: a double of the test train split test size
    Returns:
        classifier: trained classifier based on training data/labels
    '''
    counts = vectorizer.fit_transform(tweets_train.values)
    classifier = MultinomialNB()
    target = party_train.values
    classifier.fit(counts, target)
    # Writing model to a pickle to save it
    # (only for test size of 0.25)
    if test_size == 0.25:
        save_model(classifier, vectorizer)
    return classifier


def save_model(classifier, vectorizer):
    """
    Saves classifier and vectorizer into a pickle.
    Parameters:
        classifer: naive bayes classifier object
        vectorizer: Count vectorizer object
    Returns:
        None
    """
    with open('naive_classifier.pickle', 'wb') as f:
        pickle.dump(classifier, f)
    with open('naive_vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)


def matrix_display(party_test, predictions, test_size, ax):
    '''
    This function takes true labels, predictions, test_size, and an axes
    object used as parameters, to test the naive bayes classifier,
    and creates a confusion matrix showing the accuracy of the classifier.
    Parameters:
        party_test: a pandas series of the labels of the test data
        predictions: a numpy array of the predictions of the labels
            of the test data
        test_size: a double that specifies the test size
        ax: a matplotlib axes object which matrix display will plot to
    Returns:
        None
    '''
    party_labels = ['Democrat', 'Republican']
    c_matrix = confusion_matrix(party_test, predictions, party_labels)

    # Generating plot labels and confusion matrix
    ax.set_title('test-size '
                 + '(' + str(test_size) + ')', pad=15)
    ax.set_xticklabels([''] + party_labels)
    ax.set_yticklabels([''] + party_labels, rotation=90, va='center')
    ax.set_ylabel('True Party')
    ax.set_xlabel('Models Predicted Party')
    ax.matshow(c_matrix, cmap='BuPu')

    # place text for the number of correct and incorrect predictions
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            ax.text(j, i, str(c_matrix[i, j]), ha="center",
                    va="center", color="black")


def plot_accuracy_bar(party_test, predictions, test_size, ax):
    """
    Saves and plots the accuracy bar plot of
    the test data and the political affiliation predictions.
    Parameters:
        party_test: a pandas series of the labels of the test data
        predictions: a numpy array of the predictions of the labels
            of the test data
        test_size: a double that specifies the test size
        ax: a matplotlib axes object which plot_accuracy_bar will plot to
    Returns:
        None
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

    # Plotting bars
    rects1 = ax.bar(x - width/2, correct, width, label='Correct Prediction')
    rects2 = ax.bar(x + width/2, incorrect, width,
                    label='Incorrect Prediction')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Tweets')
    ax.set_xlabel('Political affiliation')
    ax.set_title('test-size '
                 + '(' + str(test_size) + ')')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='center right')

    # Generating bar labels
    autolabel(rects1, ax)
    autolabel(rects2, ax)


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
    # reading csv to pandas dataframe
    extracted_tweets = pd.read_csv('ExtractedTweets.csv')
    # Grouping the data
    big_data = group_data(extracted_tweets)
    # Running the naive bayes model on the data
    naive_bayes(big_data)


if __name__ == "__main__":
    main()