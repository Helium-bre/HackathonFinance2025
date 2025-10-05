from transformers import pipeline
from webz import get_info, KEYWORDS
import numpy as np

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier2 = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sectors = [ "Materials", "Industrials", "Consumer Discretionary", "Healthcare", "Financials", "Information Technology", "Communication Services", "Utilities", "Real Estate", "Restauration"]
actor = "US"
KEYWORDS = ["Material","Healthcare","Financial","Technology"]
text = get_info(KEYWORDS,actor)


hypothesis = f"{actor} is successful in this context."
result = classifier(text, [hypothesis, f"{actor} is not successful in this context."])
s = classifier2(text, sectors)
for i in range(len(result)):
    r = result[i]
    print(f"{text[i]}: {r['labels'][0]} (score: {r['scores'][0]})")
    
    print(s[i].get("labels")[np.argmax(s[i].get("scores"))])
    # print(s[i])


# Final Score = score[sector] + k * score[financial]


# Overview for sector classification in Financial report
# Sentiment : Results of operations
# Hypothesis : Rentability, progress, potential for growth, likely to have increase