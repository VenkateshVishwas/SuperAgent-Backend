import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import joblib
import torch
import sys
import json

# Base directory where classify.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load tokenizer, model, and label encoder from absolute paths
tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join(BASE_DIR, "distilbert_super_agent", "tokenizer"))
model = DistilBertForSequenceClassification.from_pretrained(os.path.join(BASE_DIR, "distilbert_super_agent", "model"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "distilbert_super_agent", "label_encoder.joblib"))

model.eval()

# Read input query from command line
query = sys.argv[1]

# Tokenize and predict
inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

domain = label_encoder.inverse_transform([prediction])[0]

# Output result as JSON
print(json.dumps({ "domain": domain }))
