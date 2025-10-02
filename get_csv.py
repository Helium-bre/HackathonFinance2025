import csv
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Anchors to search for
anchors = [
    "item 1. business",
    "item 2. management",
    "item 2.",
    "item 2 management",
    "item 7. management",
]

def get_text():
    """Read first 3 lines of a CSV and return as a single string."""
    rows = []
    with open("output_2005.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 7:
                break
            rows.append(" ".join(row))  # convert row list to string
    return "\n".join(rows)

def extract_chunks(text, anchors, window=1500):
    """Extract chunks of text around given anchors."""
    chunks = []
    lower_text = text.lower()
    for anchor in anchors:
        idx = lower_text.find(anchor)
        if idx != -1:
            start = max(0, idx - window // 2)
            end = min(len(text), idx + window // 2)
            chunks.append(text[start+500:end])
    return chunks

def get_region_sector(text_to_analyze):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Define your categories
    regions = ["North America", "Europe", "Asia-Pacific", "South America", "Africa", "Middle East"]
    sectors = ["Energy", "Materials", "Industrials", "Healthcare", "Financials", "Technology", "Communication", "Utilities", "Real Estate", "Restauration"]

    # Run classification
    region_result = classifier(text_to_analyze, candidate_labels=regions)
    sector_result = classifier(text_to_analyze, candidate_labels=sectors)

    print("Region:", region_result["labels"][0], " (score:", region_result["scores"][0], ")")
    print("Sector:", sector_result["labels"][0], " (score:", sector_result["scores"][0], ")")
    
def get_feeling(text_to_feel):
    # Load tokenizer and model (Hugging Face model card: ProsusAI/finbert)
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    # Tokenize input
    inputs = tokenizer(text_to_feel, return_tensors="pt", truncation=True, padding=True)

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probs = F.softmax(outputs.logits, dim=-1)
    
    # Map labels
    labels = ["positive", "negative", "neutral"]

    # Get result
    predicted_label = labels[probs.argmax()]
    print(f"Sentiment: {predicted_label}")
    print(f"Probabilities: {dict(zip(labels, probs[0].tolist()))}")

# Example usage:
if __name__ == "__main__":
    text = get_text()
    chunks = extract_chunks(text, anchors)
    for i in range(len(chunks)):
        get_region_sector(chunks[i])
        get_feeling(chunks[i])