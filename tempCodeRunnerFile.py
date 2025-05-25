import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load your dataset (assuming JSON with "query" and "domain" fields)
data_path = "super_agent_query_dataset.json"
df = pd.read_json(data_path)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['domain'])

# Create Hugging Face Dataset
dataset = Dataset.from_pandas(df[['query', 'label']])

# Train/test split (80/20)
dataset = dataset.train_test_split(test_size=0.2)

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize function
def tokenize(batch):
    return tokenizer(batch['query'], padding=True, truncation=True)

# Tokenize datasets
dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load DistilBERT model for classification
num_labels = len(label_encoder.classes_)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    accuracy = (predictions == torch.tensor(labels)).float().mean().item()
    return {"accuracy": accuracy}

# Training arguments
training_args = TrainingArguments(
    output_dir="./distilbert_super_agent",
    # evaluation_strategy="epoch", # Removed deprecated argument
    eval_strategy="epoch", # Replaced with the correct argument
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    logging_dir="./logs",
    seed=42
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model and label encoder
model.save_pretrained("./distilbert_super_agent/model")
tokenizer.save_pretrained("./distilbert_super_agent/tokenizer")

import joblib
joblib.dump(label_encoder, "./distilbert_super_agent/label_encoder.joblib")

print("Training complete. Model and tokenizer saved to ./distilbert_super_agent/")
