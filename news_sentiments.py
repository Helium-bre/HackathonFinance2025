from transformers import pipeline
from webz import get_info, KEYWORDS
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


actor = "US"
text = get_info(KEYWORDS,actor)


hypothesis = f"{actor} is successful in this context."
result = classifier(text, [hypothesis, f"{actor} is not successful in this context."])
for i in range(len(result)):
    r = result[i]
    print(f"{text[i]}: {r['labels'][0]} (score: {r['scores'][0]})")