from transformers import AutoModel
import torch.nn as nn
from BERTweet import *
from loguru import logger

NUM_CLASSES = 2

# initialize the model weights
def init_weights(module):
    if type(module) in (nn.Linear, nn.Conv1d, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])

# defining the different model classes    
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

class BERT_2DCNN_LSTM(nn.Module):
    def __init__(self, bert_model_name=bert_model_name, cnn_out_channels=100, lstm_hidden_size=256):    
        super(BERT_2DCNN_LSTM, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name).to(device)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=cnn_out_channels, kernel_size=(3, self.bert.config.hidden_size), padding=(1, 0))
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_size, batch_first=True, dropout = 0.2)
        self.classifier = nn.Linear(lstm_hidden_size, NUM_CLASSES)
        self.apply(init_weights)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_output.last_hidden_state.permute(0, 2, 1) 
        cnn_output = self.cnn(bert_output).squeeze(3)  
        cnn_output = cnn_output.permute(0, 2, 1) 
        lstm_output, _ = self.lstm(cnn_output)
        logits = self.classifier(lstm_output)
        return logits
    
class BERT_2DCNN_LSTM_Attn(nn.Module):
    def __init__(self, bert_model_name=bert_model_name, cnn_out_channels=100, lstm_hidden_size=256):    
        super(BERT_2DCNN_LSTM_Attn, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name).to(device)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=cnn_out_channels, kernel_size=(3, self.bert.config.hidden_size), padding=(1, 0))
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_size, batch_first=True, dropout = 0.2)
        self.attention = Attention(lstm_hidden_size)
        self.classifier = nn.Linear(lstm_hidden_size, NUM_CLASSES)
        self.apply(init_weights)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_output.last_hidden_state.permute(0, 2, 1) 
        cnn_output = self.cnn(bert_output).squeeze(3)  
        cnn_output = cnn_output.permute(0, 2, 1) 
        lstm_output, _ = self.lstm(cnn_output)
        attention_output = self.attention(lstm_output)
        logits = self.classifier(attention_output)
        return logits

class BERT_2DCNN_BiLSTM(nn.Module):
    def __init__(self, bert_model_name, cnn_out_channels=100, lstm_hidden_size=256, num_classes=3):
        super(BERT_2DCNN_BiLSTM, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=cnn_out_channels, kernel_size=(3, self.bert.config.hidden_size), padding=(1, 0))
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_output.last_hidden_state.permute(0, 2, 1).unsqueeze(1) 
        cnn_output = self.cnn(bert_output).squeeze(3) 
        cnn_output = cnn_output.permute(0, 2, 1) 
        lstm_output, _ = self.lstm(bert_output)  
        logits = self.classifier(lstm_output[:, -1, :])  
        return logits

class BERT_2DCNN_BiLSTM_Attn(nn.Module):
    def __init__(self, bert_model_name, cnn_out_channels=100, lstm_hidden_size=256, num_classes=3):
        super(BERT_2DCNN_BiLSTM_Attn, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.cnn = nn.Conv2d(in_channels=1, out_channels=cnn_out_channels, kernel_size=(3, self.bert.config.hidden_size), padding=(1, 0))
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True, dropout=0.2)
        self.attention = Attention(lstm_hidden_size * 2)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_output.last_hidden_state.permute(0, 2, 1).unsqueeze(1) 
        cnn_output = self.cnn(bert_output).squeeze(3) 
        cnn_output = cnn_output.permute(0, 2, 1) 
        lstm_output, _ = self.lstm(cnn_output)
        attention_output = self.attention(lstm_output)  
        logits = self.classifier(attention_output[:, -1, :])  
        return logits

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        weights = self.attention(lstm_output).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        output = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return output

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


#run the model
config_path = '../config.json'
config, bert_model_name, device, dataset, test_dataset, tokenizer = setup_environment(config_path)

# setting the model as defined in the config.json
m = config["model_name"]
if m == "1dCNN":
    model = BERT_CNN_LSTM()
    logger.info("Using model: 1dCNN + LSTM")
elif m == "2dCNN_LSTM":
    model = BERT_2DCNN_LSTM()
    logger.info("Using model: 2dCNN + LSTM")
elif m == "2dCNN_LSTM_Attn":
    model = BERT_2DCNN_LSTM_Attn()
    logger.info("Using model: 2dCNN + LSTM + Attn")
elif m == "2dCNN_biLSTM":
    model = BERT_2DCNN_BiLSTM()
    logger.info("Using model: 2dCNN + biLSTM")
elif m == "2dCNN_biLSTM_Attn":
    model = BERT_2DCNN_BiLSTM_Attn()
    logger.info("Using model: 2dCNN + biLSTM + Attn")
else:
    model = BERT_2DCNN_BiLSTM()
    logger.info("Using model: 2dCNN + biLSTM")

#replace model with checkpoint model if checkpoint is provided and asked for
checkpoint_path = config["checkpoint_path"] 
if config["load_checkpoint"] and  os.path.exists(checkpoint_path):
    model = load_checkpoint(model, checkpoint_path).to(device)
    logger.info(f'Model loaded from checkpoint: {checkpoint_path}')

#running the model by using the functions defined in BERTWeet.py 
dataset, test_dataset, data_collator = prepare_datasets(dataset, test_dataset)
train_and_predict(model, dataset, test_dataset, data_collator)

exit(0)
