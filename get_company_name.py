import pandas as pd
import numpy as np


DATAPATH = "ret_sample.csv"
CHUNKSIZE = 100000
FILTER_COL = "excntry"
FILTER = "CAN"
COMPANY = "comp_001081_01C"

Yvar = "stock_ret"
Xvar = "date"
data = pd.read_csv(DATAPATH,chunksize = CHUNKSIZE)
data = data.get_chunk(1000000)
print(len(data))
companies = list(set(np.array(data["id"])))

companies_df = pd.DataFrame({"company":companies})
companies_df.to_csv("Company_ID.csv",index = False)

