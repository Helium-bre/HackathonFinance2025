import pandas as pd
import numpy as np

DATAPATH = "ret_sample.csv"
CHUNKSIZE = 500000
FILTER_COL = "excntry"

# index = int(sys.argv[1])


def parallel_variance(n_a, avg_a, M2_a, n_b, avg_b, M2_b):
    avg_a = avg_a/n_a
    avg_b = avg_b/n_b
    n = n_a + n_b
    delta = avg_b - avg_a
    M2 = M2_a + M2_b + delta**2 * n_a * n_b / n
    var_ab = M2 / (n - 1)
    return var_ab







Yvar = "stock_ret"
Xvar = "date"
# df = None
val = pd.DataFrame(np.zeros((3,21)),columns = ["be_me","sale_me","cash_at","ni_ar1","ni_be","gp_at","op_at","ebit_sale","at_gr1", "capx_gr1", "inv_gr1", 
    "sale_gr1", "ret_12_1", "ret_3_1", "prc_highprc_252d", "beta_60m", "ivol_capm_252d", "rvol_21d", "div12m_me", "z_score","stock_ret"])
totlen = 0
cols = ["be_me","sale_me","cash_at","ni_ar1","ni_be","gp_at","op_at","ebit_sale",
        "at_gr1","capx_gr1","inv_gr1","sale_gr1","ret_12_1","ret_3_1","prc_highprc_252d",
        "beta_60m","ivol_capm_252d","rvol_21d","div12m_me","z_score","stock_ret"]

count = np.zeros(len(cols))
sum_ = np.zeros(len(cols))
sum_sq = np.zeros(len(cols))

for chunk in pd.read_csv(DATAPATH, chunksize=CHUNKSIZE, usecols=cols):
    print("next chunk")
    # Drop NA to avoid nan-propagation
    chunk = chunk.dropna()
    vals = chunk[cols].values

    count += chunk.shape[0]
    sum_ += np.sum(vals, axis=0)
    sum_sq += np.sum(vals**2, axis=0)

# Mean
mean = sum_ / count

# Variance (unbiased)
var = (sum_sq - (sum_**2)/count) / (count - 1)
std = np.sqrt(var)

result = pd.DataFrame([mean, std], index=["mean","std"], columns=cols)
result.to_csv("Zscore.csv")
print(result)

 




