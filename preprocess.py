import pandas as pd
import numpy as np
import argparse
import sys

DATAPATH = "ret_sample.csv"
CHUNKSIZE = 500000
FILTER_COL = "excntry"

# index = int(sys.argv[1])

ids = pd.read_csv("Company_ID.csv")
for i in range(1000):

    id = ids.iloc[i].values[0]
    print(id)


    Yvar = "stock_ret"
    Xvar = "date"
    df = None
    for chunk in pd.read_csv(DATAPATH,chunksize = CHUNKSIZE):
        if df is None: 
            df = chunk.loc[chunk["id"] == id]
        else :
            df = pd.concat([df,chunk.loc[chunk["id"] == id]])

    df = df.drop("id",axis = 1)
    df.to_csv(f"data/{id}.csv",index = False)




