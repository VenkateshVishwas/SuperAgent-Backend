import sys
import json
from transformers import pipeline

# Fixed 10 domain labels
candidate_labels = [
    "customer_service", "education", "finance", "hr", "it_support",
    "legal", "marketing", "operations", "product_management", "sales"
]

# Read input query from command line
query = sys.argv[1]

# Load zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  # Removed device=0 for compatibility

# Classify
result = classifier(query, candidate_labels)
domain = result["labels"][0]  # top predicted domain

# Output result as JSON (flush ensures immediate output for Node)
print(json.dumps({"domain": domain}), flush=True)
