#inspiration taken from the following approaches:
#https://github.com/prateekjoshi565/twitter_sentiment_analysis/blob/master/code_sentiment_analysis.ipynb
#https://www.kaggle.com/code/stoicstatic/twitter-sentiment-analysis-for-beginners
#https://www.kaggle.com/code/asifajunaidahmad/twitter-analysis-preprocessing 


import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
from nltk.stem import PorterStemmer
import gensim
from scipy.sparse import vstack

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

f = open('../config2.json')
config = json.load(f)

file1 = config["pos_training_path"]
file2 = config["neg_training_path"]

embed_type = config["embedding_type"]
model_type = config["model_type"]

df = pd.read_fwf(file1)
df.to_csv('train_pos.csv', index=False)

df = pd.read_fwf(file2)
df.to_csv('train_neg.csv', index=False)

# Process negative tweets
with open('train_neg.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Add headers for 'id' and 'label'
data.insert(0, ['id', 'label', 'tweet'])

# Insert 'id' and 'label' for each row
for i, row in enumerate(data[1:], start=1):  # Start at 1 to skip header
    row.insert(0, '-1')  # Insert label at the beginning of the row
    row.insert(0, str(i))  # Insert ID at the beginning of the row

# Write to a new CSV file with headers
with open('train_neg_merged.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Process positive tweets
with open('train_pos.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Add headers for 'id' and 'label'
data.insert(0, ['id', 'label', 'tweet'])  

for i, row in enumerate(data[1:], start=1): 
    row.insert(0, '1')  
    row.insert(0, str(i)) 

# Write to a new CSV file with headers
with open('train_pos_merged.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Load the CSV files
df1 = pd.read_csv('train_neg_merged.csv')
df2 = pd.read_csv('train_pos_merged.csv')

# Combine the DataFrames
combined_tweets = pd.concat([df1, df2], ignore_index=True)

# Save the combined DataFrame to a CSV file
combined_tweets.to_csv('train_data.csv', index=False)
train  = pd.read_csv('train_data.csv')
print(f"combined dataframe: \n{train.head()}")

print(f"size of combined training dataframe = {train.shape[0]}\n")
dataset_size = train.shape[0]

with open(config["test_path"], 'r') as f:
    lines = f.readlines()

#remove the IDs from the test file
with open('test.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['id', 'tweet'])  # Write the header row

    for line in lines:
        tweet_id, tweet_text = line.split(',', 1)
        writer.writerow([tweet_id, tweet_text.strip()])


test=pd.read_csv('test.csv')
print(f"size of test set = {test.shape}")

plt.hist(train['tweet'].str.len(), bins=20, label="train_tweets")
plt.hist(test['tweet'].str.len(), bins=20, label="test_tweets")
plt.legend()
plt.show()

combi = train._append(test, ignore_index=True)
print(f"shape of train + test combined={combi.shape[0]}\n")

print("----Preprocessing begins----\n")

combi['clean_tweet'] = combi['tweet'].str.replace("<user>", "")
print("removed <user>\n")
combi['clean_tweet'] = combi['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
print("removed symbols\n")
combi['clean_tweet'] = combi['clean_tweet'].str.replace("<url>", "")
print("removed <url>\n")

def remove_short(tweet):
    words = tweet.split()
    filtered_words = [word for word in words if len(word) > 2]
    return ' '.join(filtered_words)

combi['clean_tweet'] = combi['clean_tweet'].apply(remove_short)
print("removed words shorter than length 2\n")
tokenized_tweet = combi['clean_tweet'].apply(lambda x: x.split())
print("tokenized all tweets\n")

stemmer = PorterStemmer()
def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]
stemmed_tweet = tokenized_tweet.apply(stem_tokens)
for i in range(len(stemmed_tweet)):
    stemmed_tweet[i] = ' '.join(stemmed_tweet[i])
    
combi['clean_tweet'] = stemmed_tweet
print("stemmed all tweets\n")

combi['clean_tweet'].to_csv("prep_data.csv", index=False)
print("----Preprocessing ends----\n")
print(combi.head())

def create_splits(train_embed, train):
    return train_test_split(train_embed, train['label'], random_state=42, test_size=config["test_size"])

def create_embeddings(embed_type, combi):
    if embed_type == "SentenceTransformer":
        print("Creating sentence embeddings\n")
        sentence_embed = SentenceTransformer("bert-base-nli-mean-tokens")
        sentence_vectors=sentence_embed.encode(combi['clean_tweet'])
        train_sent = sentence_vectors[:dataset_size,:]
        test_embed = sentence_vectors[dataset_size:,:]
        xtrain, xvalid, ytrain, yvalid = create_splits(train_sent, train)
        return xtrain, xvalid, ytrain, yvalid, test_embed
    elif embed_type == "Bow":
        print("Creating BOW vectors\n")
        bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
        bow = bow_vectorizer.fit_transform(combi['clean_tweet'])
        train_bow = bow[:dataset_size,:]
        test_embed = bow[dataset_size:,:]
        xtrain, xvalid, ytrain, yvalid = create_splits(train_bow, train)
        return xtrain, xvalid, ytrain, yvalid, test_embed
    elif embed_type == "Tfidf":
        print("Creating TF-IDF vectors\n")
        tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(combi['clean_tweet'])
        train_tfidf = tfidf[:dataset_size,:]
        test_embed = tfidf[dataset_size:,:]
        xtrain, xvalid, ytrain, yvalid = create_splits(train_tfidf, train)
        return xtrain, xvalid, ytrain, yvalid, test_embed
    else:
        print("Creating Word2Vec vectors\n")
        tokenized_tweet = combi['clean_tweet'].apply(lambda x: x.split()) 
        model_w2v = gensim.models.Word2Vec(
                    tokenized_tweet,
                    vector_size=200, 
                    window=4, 
                    min_count=2,
                    sg = 1, 
                    hs = 0,
                    negative = 8, 
                    workers= 2, 
                    seed = 31)
        model_w2v.train(tokenized_tweet, total_examples= len(combi['clean_tweet']), epochs=20)
        def word_vector(tokens, size):
            vec = np.zeros((1, size))
            valid_tokens = [model_w2v.wv[word].reshape(1, size) for word in tokens if word in model_w2v.wv]
            if valid_tokens:
                vec = np.mean(valid_tokens, axis=0)   
            return vec
        wordvec_arrays = np.zeros((len(tokenized_tweet), 200))
        for i in range(len(tokenized_tweet)):
            wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)
        wordvec_df = pd.DataFrame(wordvec_arrays)
        train_w2v = wordvec_df.iloc[:dataset_size,:]
        test_embed = wordvec_df.iloc[dataset_size:,:]
        xtrain, xvalid, ytrain, yvalid = create_splits(train_w2v, train)
        return xtrain, xvalid, ytrain, yvalid, test_embed

xtrain, xvalid, ytrain, yvalid, test_embed = create_embeddings(embed_type, combi)

def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(xvalid)
    # Check if lengths match
    if len(yvalid) != len(y_pred):
        raise ValueError(f"Length mismatch: yvalid has {len(yvalid)} samples, y_pred has {len(y_pred)} samples.")
    
    print(type(yvalid), type(y_pred))
    
    # Print the evaluation metrics for the dataset.
    print(classification_report(yvalid, y_pred))
    eval_accuracy = accuracy_score(yvalid, y_pred)
    print(f"Eval Accuracy: {eval_accuracy}")
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(yvalid, y_pred)

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted values", fontdict={'size':14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size':14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size':18}, pad=20)
    plt.show()
    return eval_accuracy

def predict(model):
    # Predict the sentiment
    sentiment = model.predict(test_embed)
    tweet_list = test['tweet'].tolist()
    id_list = test['id'].tolist()
    print(tweet_list[0])
    print(id_list[0])
    data = []
    for id_, text, pred in zip(id_list, tweet_list, sentiment):
        data.append((id_, text, pred))
    df = pd.DataFrame(data, columns=['id', 'tweet', 'label'])
    return df

def create_predictions_csv(df):
    df_selected = df[['id', 'label']]
    df_selected.to_csv('predictions_SVC.csv', index=False, header=['Id', 'Prediction'])
    print("CSV file has been saved.")

accuracies=[]

def model_training_and_prediction(model_type):
    if model_type == "LinearSVC":
        print("Linear SVC training started\n")
        SVCmodel = LinearSVC(tol=1e-5, max_iter=2000)
        SVCmodel.fit(xtrain, ytrain)
        svc_accuracy = model_Evaluate(SVCmodel)
        accuracies.append((model_type, svc_accuracy))
        print("LinearSVC prediction started\n")
        df = predict(SVCmodel)
        print(df.columns)
        create_predictions_csv(df)
    
    elif model_type == "svm-SVC":
        print("svm.SVC training started\n")
        svc = svm.SVC(kernel='linear', C=1, probability=True, max_iter=5000, verbose=1).fit(xtrain, ytrain)
        prediction = svc.predict_proba(xvalid)
        prediction_int = prediction[:,1] >= 0.3
        prediction_int = prediction_int.astype(np.int64)
        svc_accuracy = accuracy_score(yvalid, prediction_int)
        accuracies.append((model_type, svc_accuracy))
        print("svm.SVC prediction started\n")
        test_pred = svc.predict_proba(test_embed)
        test_pred_int = test_pred[:,1] >= 0.3
        test_pred_int = test_pred_int.astype(np.int64)
        test['label'] = test_pred_int
        submission = test[['id','label']]
        submission.to_csv('sub_svc.csv', index=False)

    elif model_type == "LR":
        print("LR training started\n")
        LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
        LRmodel.fit(xtrain, ytrain)
        lr_accuracy = model_Evaluate(LRmodel)
        accuracies.append((model_type, lr_accuracy))
        print("LR prediction started\n")
        df = predict(LRmodel)
        print(df.columns)
        create_predictions_csv(df)

    elif model_type == "RF":
        print("RF training started\n")
        rf = RandomForestClassifier(n_estimators=400, random_state=11, max_depth=100).fit(xtrain, ytrain)
        prediction = rf.predict(xvalid)
        rf_accuracy = accuracy_score(yvalid, prediction)
        accuracies.append((model_type, rf_accuracy))
        print("RF prediction started\n")
        test_pred = rf.predict(test_embed)
        test['prediction'] = test_pred
        submission = test[['id','prediction']]
        submission.to_csv('sub_rf.csv', index=False)

    else:
        print("Bernoulli training started\n")
        BNBmodel = BernoulliNB(alpha=2)
        BNBmodel.fit(xtrain, ytrain)
        lr_accuracy = model_Evaluate(BNBmodel)
        accuracies.append((model_type, lr_accuracy))
        print("LR prediction started\n")
        df = predict(BNBmodel)
        create_predictions_csv(df)

model_training_and_prediction(model_type)
#Saves all the eval accuracies in both the approaches to a txt file in the order that they were executed
with open("eval_accuracies.txt", "w") as file:
    for acc in accuracies:
        file.write(f"{acc}\n")
