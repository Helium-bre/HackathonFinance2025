import torch
import numpy as np
import pandas as pd
import base
from LSTM import LSTM_seq2one, LSTM_M2M, SAVENAME


# comp_210956_01W  
# comp_019141_01W

MODEL = SAVENAME
model_dict = torch.load(f"model/{MODEL}.pth")

model = LSTM_seq2one(21,64,3,1)
model.load_state_dict(model_dict)

def predict(id, date):
    df = pd.read_csv(f"data/{id}.csv")
    
    to_pred = df.index[df["date"] == date].tolist()[-1]
    Ygt = df.iloc[to_pred]["stock_ret"]
    to_use = df.iloc[to_pred - 38 : to_pred]
    # print(to_use)
    X,Y = base.get_tensor_seq2one(to_use,base.Y_pred,base.X_COL,36,wavelet = 1)
    X = X.type(torch.float32)
    return model(X) * base.zscoredf["stock_ret"].iloc[1] + base.zscoredf["stock_ret"][0],Ygt   

if __name__ == "__main__":
    print(predict("comp_019141_01W",20210331))