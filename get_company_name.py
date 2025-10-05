import pandas as pd
import numpy as np


DATAPATH = "ret_sample.csv"
CHUNKSIZE = 100000
FILTER_COL = "excntry"
FILTER = "CAN"
COMPANY = "comp_001081_01C"

Yvar = "stock_ret"
Xvar = "date"

companies = []



for chunk in pd.read_csv(DATAPATH,chunksize = CHUNKSIZE):
    companies = np.asarray(list(set(np.concat([np.asarray(chunk["id"]),companies]))))
    print(companies)
    print("chunk done")


companies_df = pd.DataFrame({"company":companies})
companies_df.to_csv("Company_ID.csv",index = False)