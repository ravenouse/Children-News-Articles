import csv
import sys
import torch
import wandb
import numpy as np
import pandas as pd
import transformers
from pathlib import Path
from nlp import load_dataset
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, AdamW, Trainer, TrainingArguments, EarlyStoppingCallback


def read_files(filename):
  grade_labels = []
  label_mappings = {'k1':0, 'g2': 1, 'g34': 2, 'g56': 3,
                    'dg_g1': 4, 'dg_g2': 5, 'dg_g3': 6,
                    'dg_g4': 7, 'dg_g5': 8, 'dg_g6': 9}
  texts = []
  with open(filename) as tf:
    reader = csv.reader(tf, delimiter='\t')
    next(reader)
    for row in reader:
      texts.append(row[-2])
      grade_labels.append(label_mappings[row[-1]])
  print(set(grade_labels))
  return texts, grade_labels

train_texts, train_labels = read_files('tk_train.csv')
#print(train_utts[:5])
dev_texts, dev_labels = read_files('tk_dev.csv')

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
dev_encodings = tokenizer(dev_texts, truncation=True, padding=True)

class TopicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TopicDataset(train_encodings, train_labels)
dev_dataset = TopicDataset(dev_encodings, dev_labels)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

config = transformers.AutoConfig.from_pretrained('roberta-base')
config.num_labels = 10
model = transformers.AutoModelForSequenceClassification.from_pretrained('roberta-base', config=config)
model.to(device)
model.train()

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

batch_size = 16
training_args = TrainingArguments(
    output_dir='roberta-topic',          # output directory
    num_train_epochs=20,              # total number of training epochs
    per_device_train_batch_size=batch_size,  # batch size per device during training
    per_device_eval_batch_size=batch_size,   # batch size for evaluation
    warmup_steps=50,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=5,
    report_to='wandb',
    run_name='roberta',
    metric_for_best_model = 'eval_f1',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,                         # the instantiated ? Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset, 
    compute_metrics=compute_metrics, 
    callbacks = [EarlyStoppingCallback(early_stopping_patience=20)]
)

trainer.train()
wandb.finish()

model.eval()
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
criterion = torch.nn.CrossEntropyLoss()
gold_labels = []
total_pred = []
total_gold = []
loss = 0

for i, batch in enumerate(dev_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    #print('Number and type of labels: ', len(labels), labels)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits

    pred = outputs[1]
    preds = torch.argmax(pred, dim=-1)

    for j, class_label in enumerate(preds):
        predicted = preds[j].item()
        gold = labels[j].item()
        total_pred.append(predicted)
        total_gold.append(gold)

dev_acc = accuracy_score(total_gold, total_pred)

print(dev_acc)
print(classification_report(total_gold, total_pred))
