import json
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

from loguru import logger
import torch
from utils import compute_metrics, read_datasets
from scipy.special import softmax

def setup_environment(config_path): 
    f = open('./config.json')
    config = json.load(f)
    bert_model_name = config["bert_model_name"]
    # print(bert_model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device {device}')
    print(torch.version.cuda)

    dataset, test_dataset = read_datasets(config)
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    logger.info(f"Tokenizer created: {tokenizer.vocab_size}" )
    return config, bert_model_name, device, dataset, test_dataset, tokenizer

def tokenize_data(example):
    return tokenizer(example['text'], truncation=True, max_length=52)

def prepare_datasets(dataset, test_dataset):
    dataset = dataset.map(tokenize_data, batched=True)
    test_dataset = test_dataset.map(tokenize_data, batched=True)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return dataset, test_dataset, data_collator

def train_and_predict(model, dataset, test_dataset, data_collator):
    logging_steps = len(dataset['train']) // config["batch_size"]
    print(logging_steps)

    early_stop = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        eval_strategy="steps",
        disable_tqdm=False,
        logging_steps=logging_steps,
        logging_strategy="steps",
        save_strategy="steps",
        warmup_ratio=0.1,
        metric_for_best_model="eval_accuracy",
        eval_steps=logging_steps // config["eval_freq"],
        save_steps=logging_steps // config["eval_freq"],
        load_best_model_at_end=True
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stop]
    )

    logger.info('Started training')
    trainer.train()
    logger.info('Ended training')

    results = trainer.predict(test_dataset)
    logits = softmax(results.predictions, axis=1)
    probabilities = softmax(results.predictions, axis=1)[:,1]
    df = pd.DataFrame(probabilities, columns=["Prediction"])
    df.index.name = "Id"
    df.index += 1
    df.to_csv(os.path.join(config["output_dir"], "probabilities.csv"))


    y_preds = np.argmax(results.predictions, axis=1)
    print(len(y_preds))

    y_preds = [-1 if val == 0 else 1 for val in y_preds]
    df = pd.DataFrame(y_preds, columns=["Prediction"])
    df.index.name = "Id"
    df.index += 1
    df.to_csv(os.path.join(config["output_dir"], "final_predictions.csv"))


config_path = './config.json'
config, bert_model_name, device, dataset, test_dataset, tokenizer = setup_environment(config_path)

dataset, test_dataset, data_collator = prepare_datasets(dataset, test_dataset)

model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=2).to(device)

#replace model with checkpoint model if checkpoint is provided and asked for
checkpoint_path = config["checkpoint_path"] 
if config["load_checkpoint"] and os.path.exists(checkpoint_path):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(device)
    logger.info(f'Model loaded from checkpoint: {checkpoint_path}')

train_and_predict(model, dataset, test_dataset, data_collator)
exit(0)

