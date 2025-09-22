import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
#data = data.loc[data["id"] == COMPANY]
# data = data.loc[data[FILTER_COL] == FILTER]
print(data.columns)

print(len(set(np.array(data["id"])) ))
# data_preview = data.iloc[0:10000]
# data_preview.to_csv("sample_preview.csv",index = False)
# plt.plot( data[Xvar],data[Yvar])

plt.plot(pd.to_datetime(data[Xvar],format = "%Y%m%d"), data[Yvar])
plt.ylabel(Yvar)
plt.xlabel(Xvar)
plt.title(f"company: {COMPANY}")
plt.savefig("test.png")