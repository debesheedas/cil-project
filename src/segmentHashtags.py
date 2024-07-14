import re
from wordsegment import load, segment
import numpy as np

def process_hashtags(tweets):
    load()
    processed_tweets = []
    for tweet in tweets:
        processed_tweet = tweet
        found = re.findall(r"#(\w+)", tweet)
        if(len(found) > 0):
            processed_tweet = processed_tweet.replace('#', '')
            for x in found:
                segmented = " ".join(segment(x))
                processed_tweet = processed_tweet.replace(x, segmented)
        processed_tweets.append(processed_tweet)
    return np.array(processed_tweets)