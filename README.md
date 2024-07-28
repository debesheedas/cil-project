# Text Classification 2024

## Computational Intelligence Laboratory Project 
- Debeshee
- Laura
- Mariia
- Piyushi

Our text classification pipeline is broken into the following segments for modularity:
1. Data Analysis and Vizualization
2. Data Preprocessing
3. Training Model and Generating Predictions
4. Combining Predictions

Throughout our systematic and research oriented approach, we tried out various preprocessing techniques, models - each with various hyperparameter configurations. Hence, we decided to breakup the pipeline as above, allowing us to combine predictions generated using different methods into a final prediction in Step 4. 

## Setup Instructions

Create a virtual environment if you desire, and then run the following command in the home directory to install all the required packages. There are two parts in this file - one set of packages for the classical method code and the other set of packages for the deep learning approach. 

```pip install -r requirements.txt```

## Data Analysis

Run the analysis.ipynb file to see a vizualization of the dataset and get an idea about the distribution of hashtags, emoticons, etc.

## Preprocessing

The neg-pos.py file consists of all the preprocessing discussed in the paper. Apart from the default settings, other options can be enabled using the flags. For example, if you want to replace the abbreviations with their full forms, you only need to set the ABBREV flag to True and then generate the preprocessed dataset. The paths to the preprocessed dataset and test data are already added to the config file. segmentHashtags.py is a separate file that is used to break up the words in a hashtag to take that sentiment into account.



## Baselines

- Classical Methods:

We begin this project by trying out the classical algorithms such as SVC, Logistic Regression, Bernoulli and Random Forest. Different types of embeddings are used such as Bag of Words, TF-IDF, Word2Vec and Sentence Embeddings. classical.py has all the models and respective training snippets using all types of embeddings. All other models can be commented out apart from the desired model and embedding type. The accuracies are appended to a txt file and the predictions are saved to a csv file. Moreover, for large datasets (with 1M tweets) batch processing is also included.

- BERTweet:

Since the scores from the classical methods do not cross the baseline score on the leaderboard, we turn to models fine-tuned for Twitter sentiment classification. BERTweet is the state-of-the-art which crosses the baseline without any preprocessing. BERTweet.py is the mere implementation of this architecture. The config file can be used to change the dataset paths.

## BERTweetConvFusionNet

BERTweet_extended.py has the novel architecture that we propose in our project. It derives some inspiration from an existing architecture where CNN and LSTM layers are stacked upon the BERTweet embeddings. Our novel introduction is the addition of an Attention layer. We also try out both Bi-LSTM and LSTM alongwith 1D and 2D CNN layers. 
You can choose between the following options for the novel architecture and set these strings in the config.json:
- 1dCNN_LSTM: to use the BERTweet embeddings with a 1D-CNN and uni-directional LSTM
- 2dCNN_LSTM: to use the BERTweet embeddings with a 2D-CNN and uni-directional LSTM
- 2dCNN_biLSTM: to use the BERTweet embeddings with a 2D-CNN and bi-directional LSTM
- 2dCNN_LSTM_Attn: to use the BERTweet embeddings with a 2D-CNN and uni-directional LSTM, followed by an attention layer
- 2dCNN_biLSTM_Attn: to use the BERTweet embeddings with a 2D-CNN and bi-directional LSTM, followed by an attention layer
If nothing is initialized, 2dCNN_biLSTM is used. 

Once we obtain the probabilites for each model, we run an ensembling using ensemble.py. To achieve the best score, load the probabilities.csv of all the models into the config (you have to change the paths to match your folder structure).
