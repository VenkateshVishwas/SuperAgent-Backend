import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import joblib

# Load tokenizer, model, and label encoder
tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert_super_agent/tokenizer")
model = DistilBertForSequenceClassification.from_pretrained("./distilbert_super_agent/model")
label_encoder = joblib.load("./distilbert_super_agent/label_encoder.joblib")

# Put model in eval mode
model.eval()

# Loop to allow multiple queries
print("Type a query to test (or type 'exit' to quit):")
while True:
    query = input(">> ")
    if query.lower() in ("exit", "quit"):
        break

    # Tokenize input
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)

    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    # Decode predicted label
    domain = label_encoder.inverse_transform([prediction])[0]
    print(f"🔍 Routed to agent: {domain}\n")
