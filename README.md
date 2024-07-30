# BERTweetConvFusionNet: Enhancing BERTweet for Twitter Text Sentiment Classification

## Computational Intelligence Laboratory Project 

We propose a novel deep-learning based solution for Twitter
sentiment analysis that addresses the challenges of automatic
and noisily labelled data. Leveraging the pre-trained BERTweet
model [[1]](#1) for embeddings, we develop a novel CRNN-based ‘fusion
net’ architecture combining CNN, RNN, and Attention layers.
Through a systematic exploration of various neural network,
pre-processing, and ensembling configurations, our approach
achieves an accuracy of 92.12% on the public leaderboard. Details of our approach and results can be found in our [Report](./Enhancing_BERTweet_for_Twitter_Sentiment_Classification.pdf).

## Implementation

Our text classification pipeline is broken into the following segments for modularity:
1. Data Analysis and Vizualization
2. Data Preprocessing
3. Training Model and Generating Predictions
4. Combining Predictions

Throughout our systematic and research oriented approach, we tried out various preprocessing techniques, models - each with various hyperparameter configurations. Hence, we decided to breakup the pipeline as above, allowing us to combine predictions generated using different methods into a final prediction in Step 4. 

## Setup Instructions

Create a virtual environment if you desire, and then run the following command in the home directory to install all the required packages. 

```pip install -r requirements.txt```

## Data Analysis

Run the [analysis.ipynb](./src/1-Classical_Methods/analysis.ipynb) notebook to see a vizualization of the dataset and get insights about the distribution of hashtags, emoticons, etc.

## Preprocessing

The [neg-pos.py](./src/2-Preprocessing/neg-pos.py) file consists of all the preprocessing discussed in the paper. In addition to the basic pre-processing settings, other options can be enabled using the flags. Following is the list of flag names that you can add in the ``preproc`` list in the config. Please copy-paste the exact string in the code to prevent typos. The paths to the original training data, the new preprocessed dataset and test data should be added to the config as shown in the snippet below.

Flag names:
```
- HASHTAG_SEGM
- USR
- ABBREV
- EMOJI
- STEM
```

```
    "preproc":["ADD LIST OF PREPROCESSING FLAGS"],
    "neg_prep_path": <PATH TO NEGATIVE TRAINING SET>,
    "pos_prep_path": <PATH TO POSITIVE TRAINING SET>,
    "test_prep_path": <PATH TO TEST SET>",
    "neg_training_path": <PATH TO SAVE THE NEW PREPROCESSED NEGATIVE TRAINING SET>
    "pos_training_path": <PATH TO SAVE THE NEW PREPROCESSED POSITIVE TRAINING SET>
    "test_path": <PATH TO SAVE THE PREPROCESSED TEST SET>
```
After modifying the config, make sure to navigate into the [``./src/2-Preprocessing``](./src/2-Preprocessing) folder before running the following command to generate the pre-trained dataset.

```python3 neg-pos.py```

## Baselines

### Classical ML Methods:

We begin this project by trying out the classical algorithms such as SVC, Logistic Regression, Bernoulli and Random Forest. Different types of embeddings are used such as Bag of Words, TF-IDF, Word2Vec and Sentence Embeddings. [classical.py](./src/1-Classical_Methods/classical.py) contains all the models and respective training snippets. In order to choose a specific embedding type and model, you can edit the ``embedding_type`` and ``model_type`` parameteres respectively in config. You will also have to specify the paths to the training and test sets as shown below in the config. We use a flag-based logic to enable an embedding and model of choice. Pleaes copy-paste the exact string provided in the list below to avoid any failures in the code. The accuracies are appended to a txt file and the predictions are saved to a csv file.

```
embedding_type:
- SentenceTransformer
- Bow
- Tfidf
- W2v
```
```
model_type:
- LinearSVC
- svm_SVC
- LR
- RF
- BNB
```

```
    "embedding_type": "<EMBEDDING NAME>",
    "model_type": <MODEL NAME>,
    "neg_training_path": <PATH TO NEGATIVE TRAINING SET>
    "pos_training_path": <PATH TO POSITIVE TRAINING SET>
    "test_path": <PATH TO TEST SET>
```
After modifying the config, make sure to navigate into the [``./src/1-Classical_Methods``](./src/1-Classical_Methods) folder before running the following command to train the model and generate predictions.

```python3 classical.py```

### Deep Learning Based - BERTweet:

For our deep learning baselines, we fine-tune the BERTweet model [[1]](#1).
The script for the same is provided in [BERTweet.py](./src/3-Models/BERTweet.py).
To train the model and generate predictions with early stopping criteria, modify the following in [``config.json``](./src/config.json) to adjust training parameters and data and output folders accordingly:

```
    "output_dir": <PATH TO SAVE MODEL CHECKPOINTS AND FINAL PREDICTIONS>,
    "neg_training_path": <PATH TO NEGATIVE TRAINING SET>
    "pos_training_path": <PATH TO POSITIVE TRAINING SET>
    "test_path": <PATH TO TEST SET>
    "load_checkpoint": false, --- set to true if loading from a checkpoint and set path in "checkpoint_path"
    "checkpoint_path": "", --- Provide path to checkpoint folder if restarting training from a checkpoint
    "eval_freq": 5, --- no of time validation accuracy is computed per epoch
    "test_size": 0.1 --- size of validation dataset
```
After modifying the config, make sure to navigate into the [``./src/3-Models``](./src/3-Models) folder before running the following command.

```python3 BERTweet.py```

The final prediction probabilities and labels will be saved in the output directory specified in the [``config.json``](./src/config.json) as ``probabilities.csv`` and ``final_predictions.csv`` respectively. 
To fine-tune the model using a pre-processed dataset, simply change the corresponding paths to the training and test set in the [``config.json``](./src/config.json).


## BERTweetConvFusionNet

_BERTweetConvFusionNet_, our novel architecture, is a Convolutional Recurrent Neural Network or CRNN-based 'fusion net', notable for its ability to to capture local and sequential patterns and context in the text data, leading to better performance in sentiment classification. Building on previous work [[2]](#2), our novel introduction is the addition of an Attention layer. We also try out both Bi-LSTM and LSTM alongwith 1D and 2D CNN layers. 
Various configuations can be built and fine-tuned using [BERTweet_extended.py](./src/3-Models/BERTweet_extended.py). 

![Picture of architecture of BERTweetConvFusionNet](https://github.com/user-attachments/assets/4bf00797-694a-4293-86f3-6b748565fb40)


You can choose between the following options for the novel architecture and set these strings in the [``config.json``](./src/config.json). The default model used is a ``2dCNN_biLSTM``. All other model parameters such as ``epoch`` and ``batch_size`` can be modified similarly [BERTweet.py](./src/3-Models/BERTweet.py).
```
"model_name": "2dCNN_biLSTM" 
```

- ``1dCNN_LSTM``: to use the BERTweet embeddings with a 1D-CNN and uni-directional LSTM
- ``2dCNN_LSTM``: to use the BERTweet embeddings with a 2D-CNN and uni-directional LSTM
- ``2dCNN_biLSTM``: to use the BERTweet embeddings with a 2D-CNN and bi-directional LSTM
- ``2dCNN_LSTM_Attn``: to use the BERTweet embeddings with a 2D-CNN and uni-directional LSTM, followed by an attention layer
- ``2dCNN_biLSTM_Attn``: to use the BERTweet embeddings with a 2D-CNN and bi-directional LSTM, followed by an attention layer

After modifying the config, make sure to navigate into the [``./src/3-Models``](./src/3-Models) folder before running the following command to train the model and generate predictions.

```python3 BERTweet_extended.py```

## Ensembling

We provide an ensembling script in [ensemble.py](./src/4-Ensemble/ensemble.py) to ensemble the final predicted probabilities of two or more models. Before running this script, simply add the paths of all the files with the probabilities in the [``config.json``](./src/config.json) as shown below:

```
"prediction_paths": [
    <PATH TO PROBABILITIES_FROM_MODEL_1.csv>,
    <PATH TO PROBABILITIES_FROM_MODEL_2.csv>,
    ...
]
```
After modifying the config, make sure to navigate into the [``./src/4-Ensemble``](./src/4-Ensemble) folder before running the following command to generate the final ensembled predictions.

```python3 ensemble.py```


### Reproducing the best classification accuracy
To achieve the best score, load the probabilities.csv of all the models into the config (you have to change the paths to match your folder structure).

Example config for the best performing ensemble which consisted of the following models:
1. Fine-tuned BERTweet without pre-processing
2. Fine-tuned BERTweet with best pre-processing (Basic + emoticon pre-processing)
3. BERTweetConvFusionNet with `2dCNN_biLSTM` configuration
4. BERTweetConvFusionNet with `2dCNN_biLSTM_Attn` configuration
```
"prediction_paths": [
    "/home/laschulz/cil-project/data/final_table/BERT_2DCNN_BiLSTM_Attn/prob.csv",
    "/home/laschulz/cil-project/data/final_table/no_pre/prob.csv",
    "/home/laschulz/cil-project/data/final_table/bertweet_best_preprocessing/prob.csv",
    "/home/laschulz/cil-project/data/final_table/BERT_2DCNN_BiLSTM/prob.csv"
]
```


### Contributors:
- Debeshee
- Laura
- Mariia
- Piyushi




## References
<a id="1">[1]</a> 
Nguyen, Dat Quoc, Thanh Vu, and Anh Tuan Nguyen. "BERTweet: A pre-trained language model for English Tweets." arXiv preprint arXiv:2005.10200 (2020).


<a id="2">[2]</a> 
Kokab, Sayyida Tabinda, Sohail Asghar, and Shehneela Naz. "Transformer-based deep learning models for the sentiment analysis of social media data." Array 14 (2022): 100157.
