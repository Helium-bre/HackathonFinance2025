import numpy as np
import pandas as pd
import torch



X_COL = ["be_me","ep","sale_me","cash_at","ni_ar1","ni_be","gp_at","op_at","ebit_sale","at_gr1", "capx_gr1", "inv_gr1", 
        "sale_gr1", "ret_12_1", "ret_3_1", "prc_highprc_252d", "beta_60m", "ivol_capm_252d", "rvol_21d", "div12m_me", "z_score"]
ALL_COL = X_COL + ["stock_ret"]
PATHS = ["data/comp_274814_01W.csv","data/crsp_59459.csv","data/comp_210956_01W.csv","data/comp_204251_01W.csv",
         "data/crsp_81028.csv","data/comp_256545_01W.csv","data/comp_274814_01W.csv","data/crsp_92648.csv","data/comp_222763_01W.csv"]
Y_pred = "stock_ret"
def get_tensor(path : str,y : str ,x : list ,length : int):
    """ Convert csv files into formatted Tensors, and use da sliding windo"""
    df = pd.read_csv(path)
    df.fillna(0)
    y = np.asarray(df.filter([y]))
    x = np.asarray(df.filter(x))


    if len(x) < length:
        print(path)
        raise ValueError('time series is smaller than window size')

    X = torch.from_numpy(np.array([x[i:length + i,:] for i in range(len(x) - length)]))
    Y = torch.from_numpy(np.array([y[i:length + i] for i in range(len(x) - length)]))

    return X,Y

def sliding_window(x,y,length):
    """da windo be slidiin"""
    if len(x) < length:

        raise ValueError('time series is smaller than window size')

    X = torch.from_numpy(np.array([x[i:length + i,:] for i in range(len(x) - length)]))
    Y = torch.from_numpy(np.array([y[i:length + i] for i in range(len(x) - length)]))

    return X,Y

def get_dataset(path_list : list,y : str ,x : list , length : int):
    """ Return a TensorDataset of all time series with a sliding window approach of given length"""
    X = None
    Y = None
    for path in path_list : 
        Xpath,Ypath = get_tensor(path,y,x,length)
        if X is None or Y is None:
            X = Xpath
            Y = Ypath
        else :
            X = torch.cat([X,Xpath], dim = 0)
            Y = torch.cat([Y,Ypath],dim = 0)
    print(X.shape)
    print(Y.shape)
    X = X.type(torch.float32)
    Y = Y.type(torch.float32)
    return torch.utils.data.TensorDataset(X,Y)




if __name__ == "__main__":
    get_dataset(PATHS,Y_pred,X_COL,30)