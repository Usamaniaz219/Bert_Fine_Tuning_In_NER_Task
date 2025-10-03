# train.py
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import evaluate
import os

# -------------------------------
# Setup
# -------------------------------

label_list = ['O', 'SPACES', 'UNIT', 'UNIT_TYPE']
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {str(v): k for k, v in label_to_id.items()}  # str keys for JSON serialization

with open('results/checkpoint-166/id_to_label.json', 'w') as f:
    json.dump(id_to_label, f)
    print("id to label is saved ")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# -------------------------------
# Load & Preprocess Data
# -------------------------------

def load_data(json_path):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    entries = raw_data["annotations"]
    data = []
    for item in entries:
        sentence = item[0]
        entities = item[1].get("entities", [])
        char_labels = ['O'] * len(sentence)
        for start, end, label in entities:
            for i in range(start, end):
                if i < len(char_labels):
                    char_labels[i] = label

        tokenized = tokenizer(sentence, return_offsets_mapping=True, truncation=True, max_length=128)
        offsets = tokenized["offset_mapping"]

        labels = []
        for offset in offsets:
            start, end = offset
            if start == end:
                labels.append('O')
            else:
                span_labels = char_labels[start:end]
                main_label = next((l for l in span_labels if l != 'O'), 'O')
                labels.append(main_label)

        tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
        data.append((tokens, labels))
    return data

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

class ParkingDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

# -------------------------------
# Load data
# -------------------------------

data = load_data("parking_annotation_v5.json")
tokens, labels = zip(*data)

train_tokens, val_tokens, train_labels, val_labels = train_test_split(
    tokens, labels, test_size=0.2, random_state=42
)

train_encodings = tokenize_and_align_labels({"tokens": list(train_tokens), "labels": list(train_labels)})
val_encodings = tokenize_and_align_labels({"tokens": list(val_tokens), "labels": list(val_labels)})

train_dataset = ParkingDataset(train_encodings)
val_dataset = ParkingDataset(val_encodings)

# -------------------------------
# Model
# -------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(label_list))
model.to(device)

# -------------------------------
# Evaluation Metric
# -------------------------------

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=2)

    true_predictions = [
        [id_to_label[str(p.item())] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[str(p.item())] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    return seqeval.compute(predictions=true_predictions, references=true_labels)

# -------------------------------
# Training
# -------------------------------

training_args = TrainingArguments(
    output_dir="./result",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./log",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# -------------------------------
# Save model, tokenizer, and label map
# -------------------------------

trainer.save_model("/media/usama/SSD/Usama_dev_ssd/name_entity_recognition_/bert_based_NER/src/results")
tokenizer.save_pretrained("/media/usama/SSD/Usama_dev_ssd/name_entity_recognition_/bert_based_NER/src/results")

with open('results/checkpoint-166/id_to_label.json', 'w') as f:
    json.dump(id_to_label, f)
    print("id to label is saved ")

print("Training complete. Model saved to 'results'")
