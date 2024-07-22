# Text Classification 2024

## Computational Intelligence Laboratory Project 
- Debeshee
- Laura
- Mariia
- Piyushi


## Setup Instructions

Create a virtual environment if you desire, and then run the following command in the home directory to install all the required packages

```pip install -r requirements.txt```

Our text classification pipeline is broken into the following segments for modularity:
1. Data Preprocessing
2. Training Model and Generating Predictions
3. Combining Predictions

Throughout our systematic and research oriented approach, we tried out various preprocessing techniques, models - each with various hyperparameter configurations. Hence, we decided to breakup the pipeline as above, allowing us to combine prdictions generated using different methods into a final prediction in Step 3. 

You can choose between the following options for the novel architecture and set these strings in the config.json:
- 1dCNN
- 2dCNN_LSTM
- 2dCNN_LSTM_Attn

If nothing is initialized, the BaseModel is used. 