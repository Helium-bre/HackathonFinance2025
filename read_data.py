import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATAPATH = "ret_sample.csv"
CHUNKSIZE = 10000
FILTER_COL = "excntry"
FILTER = "CAN"

Yvar = "stock_ret"
Xvar = "date"
data = pd.read_csv(DATAPATH,chunksize = CHUNKSIZE)
data = data.get_chunk(10000)
data = data.loc[data[FILTER_COL] == FILTER]
print(data.columns)
print(len(data))
# data_preview = data.iloc[0:5000]
# data_preview.to_csv("sample_preview.csv",index = False)

# plt.plot( data[Xvar],data[Yvar])
plt.scatter(data[Xvar], data[Yvar])
plt.ylabel(Yvar)
plt.xlabel(Xvar)
plt.title(f"filter : {FILTER_COL}, filter value : {FILTER}")
plt.savefig("test.png")