from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch


model_name = "ProsusAI/finbert"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


finbert_pipeline = pipeline("sentiment-analysis", 
                           model=model, 
                           tokenizer=tokenizer)


financial_text = "The company had unpredicted crisis, leading to a decrease in stock value"
result = finbert_pipeline(financial_text)
print(result)