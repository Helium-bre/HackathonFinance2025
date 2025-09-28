from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pickle
import pandas as pd
model_name = "ProsusAI/finbert"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

key = "cfyd"
key_value = 300158

finbert_pipeline = pipeline("sentiment-analysis", 
                           model=model, 
                           tokenizer=tokenizer) 


financial_text = "positive and negative"  # NEED TO TRUNCATE IT (partition into smaller text). finBERT does not handle big texts. 

with open('text_us_2006.pkl','rb') as file:
    data = pickle.load(file)

df = pd.DataFrame(data)
print(df.loc[df[key] == key_value]["mgmt"])
# print(financial_text)
# result = finbert_pipeline(financial_text)
# print(result)