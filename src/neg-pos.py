import pandas as pd
import json
import json
import re
from segmentHashtags import *
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

HASHTAG_SEGM=True
USR=False
ABBREV=False
EMOJI=True
STEM = False

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'silly', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}

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


    #combined_df = combined_df.drop_duplicates(subset='tweet', keep=False)
    pos_unique_df = combined_df[combined_df['label'] == 'positive'].drop(columns=['label'])
    neg_unique_df = combined_df[combined_df['label'] == 'negative'].drop(columns=['label'])

    # duplicates = combined_df.duplicated(subset='tweet', keep=False)
    # unique_df = combined_df[~duplicates]
    # unique_df = combined_df
    # unique_df = combined_df
    # neg_unique_df = unique_df[unique_df['label'] == 'negative'].drop(columns=['label'])

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
    if USR: modified_tweets = [tweet.replace("<user>", "@USER") for tweet in tweets]
    else: modified_tweets = [tweet.replace("<user>", "") for tweet in tweets]
    modified_tweets = [tweet.replace("<url>", "HTTPURL") for tweet in modified_tweets]
    if ABBREV: modified_tweets = [replace_abbreviation(x, abbreviations) for x in modified_tweets]
    if EMOJI: modified_tweets = [replace_emojis(x, emojis) for x in modified_tweets]
    if STEM: modified_tweets = [stem_tweet(x) for x in modified_tweets]
    if ABBREV: modified_tweets = [replace_abbreviation(x, abbreviations) for x in modified_tweets]
    if EMOJI: modified_tweets = [replace_emojis(x, emojis) for x in modified_tweets]
    if STEM: modified_tweets = [stem_tweet(x) for x in modified_tweets]
    # modified_tweets = [tweet.replace("<user>", "") for tweet in tweets]
    # modified_tweets = [tweet.replace("<url>", "") for tweet in modified_tweets]
    modified_tweets = [tweet.replace("[^a-zA-Z#]", "") for tweet in modified_tweets]
    modified_tweets = [tweet.strip() + '\n' for tweet in modified_tweets]
    write_tweets(filename_new, modified_tweets)
    print(f"Tags replaced and tweets cleaned in {filename_new}")


f = open('./config.json')
config = json.load(f)
# File paths
positive_tweets_file = config["pos_prep_path"]
negative_tweets_file = config["neg_prep_path"]
# pos_unique_file = "/home/pgoyal/cil-project/data/pos_unique_tweets.txt"
# neg_unique_file = "/home/pgoyal/cil-project/data/neg_unique_tweets.txt"
pos_unique_file = config["pos_training_path"]
neg_unique_file = config["neg_training_path"]
test_file = config["test_prep_path"]

# Remove common tweets and write unique tweets to new files
find_and_remove_common_tweets(positive_tweets_file, negative_tweets_file, pos_unique_file, neg_unique_file)

# Replace tags in the unique tweets files
replace_tags(pos_unique_file, pos_unique_file)
replace_tags(neg_unique_file, neg_unique_file)
replace_tags(test_file, config["test_path"])
replace_tags(test_file, config["test_path"])