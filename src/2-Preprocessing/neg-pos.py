import pandas as pd
import json
import re
from segmentHashtags import *
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
f = open('../config.json')
config = json.load(f)

HASHTAG_SEGM=False
USR=False
ABBREV=False
EMOJI=True
STEM = False

prepoc_list = config["preproc"]
if "HASHTAG_SEGM" in prepoc_list:
    HASHTAG_SEGM=True
if "USR" in prepoc_list:
    USR=True
if "ABBREV" in prepoc_list:
    ABBREV=True
if "EMOJI" in prepoc_list:
    EMOJI=True
if "STEM" in prepoc_list:
    STEM=True

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'silly', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

def load_abbreviations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        abbreviations = json.load(file)
    return abbreviations

abbreviations = load_abbreviations('./abbreviations.json')

def replace_abbreviation(tweet, abbreviations):
    for word in abbreviations.keys():
        tweet = re.sub(r'\b' + re.escape(word) + r'\b', abbreviations[word], tweet)
    return tweet

def replace_emojis(tweet, emojis):
    for word in emojis.keys():
        tweet = re.sub(r'\b' + re.escape(word) + r'\b', emojis[word], tweet)
    return tweet

def stem_tweet(tweet):
    tokens = tweet.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def read_tweets_to_df(filename, label):
    with open(filename, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    return pd.DataFrame({'tweet': tweets, 'label': label})

def write_tweets_from_df(filename, df):
    with open(filename, 'w', encoding='utf-8') as file:
        file.writelines(df['tweet'].tolist())

def find_and_remove_common_tweets(pos_file, neg_file, pos_unique_file, neg_unique_file):
    pos_df = read_tweets_to_df(pos_file, 'positive')
    neg_df = read_tweets_to_df(neg_file, 'negative')
    combined_df = pd.concat([pos_df, neg_df])

    duplicates_grouped=combined_df.groupby(['tweet', 'label']).size().reset_index(name='counts')
    duplicates_grouped=duplicates_grouped.sort_values(['tweet', 'counts'], ascending=False)
    we_keep=duplicates_grouped.drop_duplicates(subset='tweet', keep='first')
    we_keep = we_keep.drop(columns=['counts'])

    combined_df = pd.merge(combined_df, we_keep, on=['tweet', 'label'], how='inner')

    pos_unique_df = combined_df[combined_df['label'] == 'positive'].drop(columns=['label'])
    neg_unique_df = combined_df[combined_df['label'] == 'negative'].drop(columns=['label'])

    write_tweets_from_df(pos_unique_file, pos_unique_df)
    write_tweets_from_df(neg_unique_file, neg_unique_df)

def read_tweets(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.readlines()

def write_tweets(filename, tweets):
    with open(filename, 'w', encoding='utf-8') as file:
        file.writelines(tweets)

def replace_tags(filename_old, filename_new):
    tweets = read_tweets(filename_old)
    if HASHTAG_SEGM: tweets = process_hashtags(tweets)
    if USR: modified_tweets = [tweet.replace("<user>", "@USER") for tweet in tweets]
    else: modified_tweets = [tweet.replace("<user>", "") for tweet in tweets]
    if ABBREV: modified_tweets = [replace_abbreviation(x, abbreviations) for x in modified_tweets]
    if EMOJI: modified_tweets = [replace_emojis(x, emojis) for x in modified_tweets]
    if STEM: modified_tweets = [stem_tweet(x) for x in modified_tweets]
    if ABBREV: modified_tweets = [replace_abbreviation(x, abbreviations) for x in modified_tweets]
    modified_tweets = [tweet.replace("<url>", "") for tweet in modified_tweets]
    modified_tweets = [tweet.replace("[^a-zA-Z#]", "") for tweet in modified_tweets]
    modified_tweets = [tweet.strip() + '\n' for tweet in modified_tweets]
    write_tweets(filename_new, modified_tweets)
    print(f"Tags replaced and tweets cleaned in {filename_new}")

# set file paths
positive_tweets_file = config["pos_prep_path"]
negative_tweets_file = config["neg_prep_path"]
pos_unique_file = config["pos_training_path"]
neg_unique_file = config["neg_training_path"]
test_file = config["test_prep_path"]

# Remove common tweets and write unique tweets to new files
find_and_remove_common_tweets(positive_tweets_file, negative_tweets_file, pos_unique_file, neg_unique_file)

# Replace tags in the unique tweets files
replace_tags(pos_unique_file, pos_unique_file)
replace_tags(neg_unique_file, neg_unique_file)
replace_tags(test_file, config["test_path"])
