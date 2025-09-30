from transformers import pipeline
from webz import get_info, KEYWORDS
import numpy as np

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier2 = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sectors = ["Energy", "Materials", "Industrials", "Consumer Discretionary", "Healthcare", "Financials", "Information Technology", "Communication Services", "Utilities", "Real Estate", "Restauration"]
actor = "US"
text = get_info(KEYWORDS,actor)


hypothesis = f"{actor} is successful in this context."
result = classifier(text, [hypothesis, f"{actor} is not successful in this context."])
s = classifier2(text, sectors)
for i in range(len(result)):
    r = result[i]
    print(f"{text[i]}: {r['labels'][0]} (score: {r['scores'][0]})")
    
    print(sectors[np.argmax(s[i].get("scores"))])
    print(s[i].get("scores"))

