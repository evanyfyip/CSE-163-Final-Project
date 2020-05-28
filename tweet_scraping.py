from bs4 import BeautifulSoup
import requests
import sys
import json
import pandas as pd
import os
import pickle
import re

def get_tweets(username):
    """
    Takes in username and returns a pandas dataframe of their
    tweets
    """
    url = 'https://twitter.com/' + username

    # Getting html
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    # locating and getting tweets
    tweets = soup.find_all("li", {"data-item-type": "tweet"})

    tweets_list = list()
    for tweet in tweets:
        #     for tweet in tweets:
        # tweet_data = None
        # try:
        #     tweet_data = get_tweet_text(tweet)
        # except Exception as e:
        #     continue
        #     #ignore if there is any loading or tweet error
        try:
            tweet_text_box = tweet.find("p", {"class": "TweetTextSize TweetTextSize--normal js-tweet-text tweet-text"})
            tweet_text = tweet_text_box.text
            images_in_tweet_tag = tweet_text_box.find_all("a", {"class": "twitter-timeline-link u-hidden"})
            # removing images
            for image_in_tweet_tag in images_in_tweet_tag:
                tweet_text = tweet_text.replace(image_in_tweet_tag.text, '')
            # removing urls
            tweet_string = str(tweet_text)
            # removing new line characters
            clean_tweet = tweet_text.replace(u'\n', u'')
            clean_tweet = re.sub(r"http\S+", "", clean_tweet) # removing links
            clean_tweet = clean_tweet.replace(u'\xa0', u' ')  # removing extra characters
            tweets_list.append({'username':username, 'tweet':clean_tweet})
        except Exception as e:
            continue
        #ignore if there is any loading or tweet error
    
    # converting to dataframe
    tweet_df = pd.DataFrame(tweets_list)
    return tweet_df

def get_all_tweets(usernames):
    """
    Takes in list of usernames and returns
    pandas dataframe with each person as a column
    """
    length = len(usernames)
    for i in range(length):
        if i == 0:
            tweets = get_tweets(usernames[i])
        else:
            new_tweets = get_tweets(usernames[i])
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
    with open('scraped_tweets.pickle', 'wb') as f:
        pickle.dump(tweets, f)
        
    # Verifying it saved a file called good_model.pickle
    print('Files in Directory:', os.listdir('.'))

if __name__ == '__main__':
    main()
