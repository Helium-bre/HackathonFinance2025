from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pickle
import pandas as pd

model_name = "ProsusAI/finbert"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

key = "gvkey"
key_value = 10391.0

finbert_pipeline = pipeline("sentiment-analysis", 
                           model=model, 
                           tokenizer=tokenizer) 
#"facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model="ProsusAI/finbert" , tokenizer =  "ProsusAI/finbert"  )

#financial_text = "positive and negative"  # NEED TO TRUNCATE IT (partition into smaller text). finBERT does not handle big texts. 

with open('text_us_2006.pkl','rb') as file:
    data = pickle.load(file)

df = pd.DataFrame(data)
print(df.columns)
print(df)
currcomp_data = df.loc[df[key] == key_value]["mgmt"].iloc[0].lower()
with open("text.txt","w") as f:
    f.write(currcomp_data)
# print(df.loc[df[key] == key_value]["mgmt"].iloc[0])
content_and_hypothesis= {"heading":["overview","results of operations","risks related to our business","cost of sales"],\
                         "hypothesis":["positive","Sales and revenue have increased","The company has taken considerable amount of risk","cost and sales of the company has decreased"],\
                        "counter_hypothesis":["negative","Sales and revenue have decreased","The company has not taken considerable amount of risk","cost and sales of the company has increased"]}

for i in range(len(content_and_hypothesis["heading"])) :
    h = content_and_hypothesis["heading"][i] 
    hyp = content_and_hypothesis["hypothesis"][i]
    conthyp = content_and_hypothesis["counter_hypothesis"][i]
    if h in currcomp_data:
        print(h)
        text = currcomp_data.split(h)[1][:1500]
        s = classifier(text, [hyp,conthyp])
        print(s)
        



# print("management commentary" in currcomp_data)
# print("result of operations" in currcomp_data)
# print("outlook" in currcomp_data)
# print("executive summary" in currcomp_data)
# print("earning highlight" in currcomp_data)
# print("cost of sales" in currcomp_data)
# print("risks related to our business" in currcomp_data)
# print("overview" in currcomp_data)
# print("a" in currcomp_data)






# print(financial_text)
# result = finbert_pipeline(financial_text)
# print(result)