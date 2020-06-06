# Evan Yip, Cooper Chia, Walker Azam
# CSE 163
# Tweet scraper (before legacy twitter update)
# Scrapes tweets using list of usernames
# 6/3/2020

from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import pickle
import re


def get_tweets(username):
    """
    Takes in username and returns a pandas dataframe of their
    tweets.
    Parameters:
        username: a string of the twitter users username
    Returns:
        tweet_df: a pandas dataframe with username and tweet as
            columns
    """
    url = 'https://twitter.com/' + username

    # Getting html
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    # locating and getting tweets
    tweets = soup.find_all("div", {"data-testid": "tweet"})
    tweets_list = list()
    for tweet in tweets:
        try:
            # Retreive tweet content
            tweet_text_box = tweet.find("p", {"class": "TweetTextSize \
                                              TweetTextSize--normal \
                                              js-tweet-text tweet-text"})
            tweet_text = tweet_text_box.text
            images_in_tweet = tweet_text_box.find_all("a", {"class":
                                                      "twitter-timeline-link\
                                                      u-hidden"})
            # removing images
            for image in images_in_tweet:
                tweet_text = tweet_text.replace(image.text, '')
            # removing new line characters
            clean_tweet = tweet_text.replace(u'\n', u'')
            # removing url links
            clean_tweet = re.sub(r"http\S+", "", clean_tweet)
            # removing extra characters
            clean_tweet = clean_tweet.replace(u'\xa0', u' ')
            # generating list of dictionaries
            tweets_list.append({'username': username, 'tweet': clean_tweet})

        # ignore if loading or tweet error
        except Exception:
            continue

    # converting to dataframe
    tweet_df = pd.DataFrame(tweets_list)
    return tweet_df


def get_all_tweets(usernames):
    """
    Takes in list of usernames and returns
    combined pandas dataframe of all the users tweets.
    Parameters:
        usernames: a list of username strings
    Returns:
        tweets: a concatenated pandas dataframe of all users and their tweets.
    """
    length = len(usernames)
    # For each username, get the tweets
    for i in range(length):
        # Creating dataframe if first user
        if i == 0:
            tweets = get_tweets(usernames[i])
        else:
            new_tweets = get_tweets(usernames[i])
            # Appending the tweets to current dataframe
            tweets = pd.concat([tweets, new_tweets])
    return tweets


def main():
    usernames = ['JayInslee', 'Grimezsz', 'realDonaldTrump',
                 'elonmusk', 'BarackObama', 'BillGates',
                 'RobertDowneyJr', 'RepDelBene', 'MayorJenny',
                 'JustinTrudeau', 'BernieSanders', 'Mike_Pence',
                 'senatemajldr', 'BorisJohnson']
    tweets = get_all_tweets(usernames)

    # Save a pickle of the value. Need to open file in write-binary mode (wb)
    # Note: filename is changed to scraped_tweets_test to prevent overwriting
    # existing scraped_tweets.pickle that contains the working dataframe.
    with open('scraped_tweets_test.pickle', 'wb') as f:
        pickle.dump(tweets, f)

    # Verifying it saved a file called good_model.pickle
    print('Files in Directory:', os.listdir('.'))


if __name__ == '__main__':
    main()
