# Text Classification 2024

## Computational Intelligence Laboratory Project 
- Debeshee
- Laura
- Mariia
- Piyushi


## Setup Instructions

Create a virtual environment if you desire, and then run the following command in the home directory to install all the required packages

```pip install -r requirements.txt```
## Data Analysis
## Preprocessing
## Baselines
Classical Methods
BERTweet
## BERTweetConvFusionNet
Our text classification pipeline is broken into the following segments for modularity:
1. Data Preprocessing
2. Training Model and Generating Predictions
3. Combining Predictions

Throughout our systematic and research oriented approach, we tried out various preprocessing techniques, models - each with various hyperparameter configurations. Hence, we decided to breakup the pipeline as above, allowing us to combine prdictions generated using different methods into a final prediction in Step 3. 


You can choose between the following options for the novel architecture and set these strings in the config.json:
- 1dCNN_LSTM: to use the BERTweet embeddings with a 1D-CNN and uni-directional LSTM
- 2dCNN_LSTM: to use the BERTweet embeddings with a 2D-CNN and uni-directional LSTM
- 2dCNN_biLSTM: to use the BERTweet embeddings with a 2D-CNN and bi-directional LSTM
- 2dCNN_LSTM_Attn: to use the BERTweet embeddings with a 2D-CNN and uni-directional LSTM, followed by an attention layer
- 2dCNN_LSTM_Attn: to use the BERTweet embeddings with a 2D-CNN and bi-directional LSTM, followed by an attention layer
If nothing is initialized, the best model (xyz) is used. 

To achieve the best score, run the following commands with the current config.json initialisation (you only have to change the paths to match your folder structure):
