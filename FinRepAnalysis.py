from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pickle
import pandas as pd
import re

def extract_first_sentences(text, section_name, n_sentences=5, occurrence=1):

    lower_text = text.lower()
    matches = [m.start() for m in re.finditer(re.escape(section_name.lower()), lower_text)]
    
    if len(matches) < occurrence:
        return None  # not enough occurrences found
    
    # Pick the Nth occurrence
    start_idx = matches[occurrence - 1]
    section_text = text[start_idx + len(section_name):].lstrip()
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', section_text.strip())
    
    return " ".join(sentences[:n_sentences])


def read_rf():
    pass
    
def read_mgmt(text : str, analysis : dict):
    score_dict = {k:0 for k in analysis.keys()}
    for heading,hypothesis in analysis.items():
            section = extract_first_sentences(text,heading)
            if section:
                score_dict[heading] = get_sentiment(section,hypothesis)
    return score_dict

def get_sentiment(text, labels):  # 3 labels

    predict = classifier(text, labels)
    s = predict.get("labels")[0]
    print(predict)

    return 1 - labels.index(s)



def read_fn(filename, date, gvkey,analysis):
    with open(filename,"rb") as f:
        df = pickle.load(f)
    filtered = df.loc[(df["date"] == date) & (df["gvkey"] == gvkey)]
    if filtered.empty:
        print("not found bro")
        return None

    mgmt = filtered.get("mgmt").values

    mgmt = mgmt[0]
    return read_mgmt(mgmt,analysis)

# model_name = "ProsusAI/finbert"  
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# key = "gvkey"
# key_value = 10391.0

# finbert_pipeline = pipeline("sentiment-analysis", 
#                            model=model, 
#                            tokenizer=tokenizer) 
# #"facebook/bart-large-mnli"
# classifier = pipeline("zero-shot-classification", model="ProsusAI/finbert" , tokenizer =  "ProsusAI/finbert"  )

# #financial_text = "positive and negative"  # NEED TO TRUNCATE IT (partition into smaller text). finBERT does not handle big texts. 

# with open('text_us_2006.pkl','rb') as file:
#     data = pickle.load(file)

# df = pd.DataFrame(data)
# print(df.columns)
# print(df)
# currcomp_data = df.loc[df[key] == key_value]["mgmt"].iloc[0].lower()
# with open("text.txt","w") as f:
#     f.write(currcomp_data)
# # print(df.loc[df[key] == key_value]["mgmt"].iloc[0])
# content_and_hypothesis= {"heading":["overview","results of operations","risks related to our business","cost of sales"],\
#                          "hypothesis":["positive","Sales and revenue have increased","The company has taken considerable amount of risk","cost and sales of the company has decreased"],\
#                         "counter_hypothesis":["negative","Sales and revenue have decreased","The company has not taken considerable amount of risk","cost and sales of the company has increased"]}

# for i in range(len(content_and_hypothesis["heading"])) :
#     h = content_and_hypothesis["heading"][i] 
#     hyp = content_and_hypothesis["hypothesis"][i]
#     conthyp = content_and_hypothesis["counter_hypothesis"][i]
#     if h in currcomp_data:
#         print(h)
#         text = currcomp_data.split(h)[1][:1500]
#         s = classifier(text, [hyp,conthyp])
#         print(s)
        



# print("management commentary" in currcomp_data)
# print("result of operations" in currcomp_data)
# print("outlook" in currcomp_data)
# print("executive summary" in currcomp_data)
# print("earning highlight" in currcomp_data)
# print("cost of sales" in currcomp_data)
# print("risks related to our business" in currcomp_data)
# print("overview" in currcomp_data)
# print("a" in currcomp_data)




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

if __name__ == "__main__":
    analysis = {"results of operations" :["positive","neutral","negative"], "risks related to our business":["positive","neutral","negative"],"cost of sales":["positive","neutral","negative"]}
    print(read_fn("text_us_2006.pkl","20060104",10391.0,analysis))

# print(financial_text)
# result = finbert_pipeline(financial_text)
# print(result)

"""
rr
import re
import csv

def get_text():

    rows = []
    with open("output_2005.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 7:
                break
            rows.append(" ".join(row))  # convert row list to string
    return "\n".join(rows)



# Example usage:
if __name__ == "__main__":
    text = get_text()
    snippet = extract_first_sentences(text, "Results of Operations", n_sentences=4, occurrence=3)
    print(snippet)
    
    """