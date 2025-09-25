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
    # df = None
    val = pd.DataFrame(np.zeros((2,21)),columns = ["be_me","sale_me","cash_at","ni_ar1","ni_be","gp_at","op_at","ebit_sale","at_gr1", "capx_gr1", "inv_gr1", 
        "sale_gr1", "ret_12_1", "ret_3_1", "prc_highprc_252d", "beta_60m", "ivol_capm_252d", "rvol_21d", "div12m_me", "z_score","stock_ret"])
    val.iloc[0] = [np.inf for _ in val.columns]
    for chunk in pd.read_csv(DATAPATH,chunksize = CHUNKSIZE):
        print("next chunk")
        for col in val.columns:
            mini = np.min(chunk[col])
            maxi = np.max(chunk[col])
            if maxi > val[col].iloc[1]:
                val.loc[1,col] = maxi
            if mini < val[col].iloc[0]:
                val.loc[0,col] = mini
        # if df is None: 
        #     df = chunk.loc[chunk["id"] == id]
        # else :
        #     df = pd.concat([df,chunk.loc[chunk["id"] == id]])

    # df = df.drop("id",axis = 1)
    # df.to_csv(f"data/{id}.csv",index = False)
    val.to_csv("MinMaxVal.csv",index = False)




