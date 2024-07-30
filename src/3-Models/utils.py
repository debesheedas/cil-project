from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset, concatenate_datasets

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def read_datasets(config):
    # Adding labels to the data
    neg_dataset = load_dataset("text", data_files=config["neg_training_path"], split='train')
    label_column = [0] * len(neg_dataset)
    neg_dataset = neg_dataset.add_column("label", label_column)

    pos_dataset = load_dataset("text", data_files=config["pos_training_path"], split='train')
    label_column = [1] * len(pos_dataset)
    pos_dataset = pos_dataset.add_column("label", label_column)
    
    test_dataset = load_dataset("text", data_files=config["test_path"], split='train')

    dataset = concatenate_datasets([neg_dataset, pos_dataset])

    return dataset, test_dataset