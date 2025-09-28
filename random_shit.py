from transformers import pipeline

# Zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "B is doing bad, unlike A"
actors = ["A", "B"]

for actor in actors:
    hypothesis = f"The sentiment towards {actor} is positive."
    result = classifier(text, [hypothesis, f"The sentiment towards {actor} is negative."])
    print(f"{actor}: {result['labels'][0]} (score: {result['scores'][0]:.2f})")