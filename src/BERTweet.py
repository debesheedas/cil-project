import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import classification_report
import logging
from loguru import logger
from tqdm import tqdm
import torch
from utils import compute_metrics
from scipy.special import softmax
# from sklearn.model_selection import train_test_split


f = open('./config.json')
config = json.load(f)
# bert_model_name = 'vinai/bertweet-base'
bert_model_name = config["bert_model_name"]
print(bert_model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f'Using device {device}')

# Adding labels to the data
neg_dataset = load_dataset("text", data_files=r'/home/debdas/cil/cil-project/data/train_neg.txt', split='train')
label_column = [0] * len(neg_dataset)
neg_dataset = neg_dataset.add_column("label", label_column)

pos_dataset = load_dataset("text", data_files=r'/home/debdas/cil/cil-project/data/train_pos.txt', split='train')
label_column = [1] * len(pos_dataset)
pos_dataset = pos_dataset.add_column("label", label_column)

test_dataset = load_dataset("text", data_files=r'/home/debdas/cil/cil-project/data/test_data.txt', split='train')

dataset = concatenate_datasets([neg_dataset, pos_dataset])
print(dataset)
print(dataset[0:5])

#---------------------------------
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
logger.info("Tokenizer created")

def tokenize_data(example):
    return tokenizer(example['text'], truncation=True)

dataset = dataset.map(tokenize_data, batched=True)
test_dataset = test_dataset.map(tokenize_data, batched=True)

print(dataset)
print(dataset[0:5])

dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(dataset['train'])
print(dataset['test'])

print("**************")
#---------------------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=2).to(device)
print(model, device)


logging_steps = len(dataset['train']) // config["batch_size"]
print(logging_steps)

training_args = TrainingArguments(
    output_dir=config["output_dir"],
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    num_train_epochs=config["epochs"],
    weight_decay=config["weight_decay"],
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    logging_strategy="steps",
    save_strategy="epoch",
    warmup_ratio=0.1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


logger.info('Started training')
trainer.train()
logger.info('Ended training')

results = trainer.predict(test_dataset)
# print(results.predictions)
logits = softmax(results.predictions, axis=1)
# print(logits)
probabilities = softmax(results.predictions, axis=1)[:,1]
# print(probabilities)
df = pd.DataFrame(probabilities, columns=["Prediction"])
df.index.name = "Id"
df.index += 1
df.to_csv("../output/probabilities.csv")

y_preds = np.argmax(results.predictions, axis=1)
print(len(y_preds))

y_preds = [-1 if val == 0 else 1 for val in y_preds]
df = pd.DataFrame(y_preds, columns=["Prediction"])
df.index.name = "Id"
df.index += 1
df.to_csv("../output/final_predictions.csv")

exit(0)

