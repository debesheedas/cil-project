import json
import logging
from loguru import logger
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from BERTweet import *

NUM_CLASSES = 2

config_path = './config.json'
config, bert_model_name, device, dataset, test_dataset, tokenizer = setup_environment(config_path)

def init_weights(module):
    if type(module) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])

class BERT_CNN_LSTM(nn.Module):
    def __init__(self, bert_model_name=bert_model_name, cnn_out_channels=100, lstm_hidden_size=256):
        super(BERT_CNN_LSTM, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name).to(device)
        self.cnn = nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_size, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden_size, NUM_CLASSES)

        self.apply(init_weights)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_output.last_hidden_state.permute(0, 2, 1)  
        cnn_output = self.cnn(bert_output).permute(0, 2, 1)  
        lstm_output, _ = self.lstm(cnn_output)
        final_output = lstm_output[:, -1, :] 
        logits = self.classifier(final_output)
        return logits

# Example of how to instantiate the model
model = BERT_CNN_LSTM()

dataset = dataset.map(tokenize_data, batched=True)
test_dataset = test_dataset.map(tokenize_data, batched=True)

dataset = dataset.train_test_split(test_size=0.2, seed=42)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_and_predict(model, dataset, test_dataset, data_collator)

exit(0)