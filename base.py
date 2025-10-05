import numpy as np
import pandas as pd
import torch
import scipy
import pywt
import glob


X_COL = ["be_me","sale_me","cash_at","ni_ar1","ni_be","gp_at","op_at","ebit_sale","at_gr1", "capx_gr1", "inv_gr1", 
        "sale_gr1", "ret_12_1", "ret_3_1", "prc_highprc_252d", "beta_60m", "ivol_capm_252d", "rvol_21d", "div12m_me", "z_score"]
print(len(X_COL))
ALL_COL = X_COL + ["stock_ret"]
# PATHS = ["data/comp_274814_01W.csv","data/crsp_59459.csv","data/comp_210956_01W.csv","data/comp_204251_01W.csv","data/comp_209856_01W.csv",
#          "data/crsp_81028.csv","data/comp_256545_01W.csv","data/comp_274181_01W.csv","data/crsp_92648.csv","data/comp_222763_01W.csv",
#          "data/comp_238449_01W.csv","data/comp_275124_01W.csv","data/comp_256752_01W.csv","data/comp_066376_01C.csv","data/comp_211917_01W.csv"]
PATHS = glob.glob("data/*.csv")

# TEST_PATHS = ["data/comp_015659_02W.csv","data/comp_101270_01W.csv","data/comp_201257_01W.csv"]
TEST_PATHS = PATHS
Y_pred = "stock_ret"
minmaxdf = pd.read_csv("MinMaxVal.csv")
zscoredf = pd.read_csv("Zscore.csv")

def Zscore(df):
    """
    Apply Zscore normalization on the values
    """

    for col in df.columns :
        values = zscoredf[col]
        df[col] = (df[col] - values.iloc[0])/values.iloc[1]
    return df


def MinMax(df):
    """
    Apply minmax on values, from min and max of dataset
    """
    for col in df.columns :
        values = minmaxdf[col]
        df[col] = (df[col] - values.iloc[0])/(values.iloc[1] - values.iloc[0])
    return df


def get_tensor(path : str,y : str ,x : list ,length : int,normalize = "zscore", wavelet = True):
    """ 
    Convert csv files into formatted Tensors, and use da sliding windo
    """
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


def get_tensor_seq2one(df : pd.DataFrame,y : str ,x : list ,length : int,normalize = "zscore", wavelet = 3):
    """ 
    Convert csv files into formatted Tensors for seq2one prediction
    df : dataframe
    y : column to  (will be added to x too)
    x : list of input columns
    length : sliding window length
    normalize : method of normalization. Default is zscore
    wavelet : wavelet feature extraction
    """
    df = df.fillna(0)
    # print(df.filter(x))
    if normalize == "zscore":
        dfy = Zscore(df.filter([y]))
        dfx = Zscore(df.filter(x))
    elif normalize == "minmax":
        dfy = MinMax(df.filter([y]))
        dfx = MinMax(df.filter(x))
    else :
        dfy = df.filter([y])
        dfx = df.filter(x)
    y_arr = np.asarray(dfy)
    x_arr = np.asarray(dfx)
    x_arr = np.hstack([x_arr,y_arr.reshape(-1,1)])
    # print(dfx)
    if len(x_arr) < length:
        #raise ValueError('time series is smaller than window size')
        print("smaller than window size")
        return torch.from_numpy(np.array([])),torch.from_numpy(np.array([]))
    if wavelet != 0 :
        wave = "db4"
        wavelet_x = np.zeros((len(x_arr),wavelet*len(x_arr[0])))
        for i in range(0,len(x_arr[0]),wavelet):
            variable = x_arr[:,i]
            coeffs = pywt.wavedec(variable, wave, level=2)
            A2, D2, D1 = coeffs  # Approximation (trend), Details (medium, fine)

            # Reconstruct each component separately
            wavelet_x[:,i] = pywt.upcoef('a', A2, wave, level=2, take=len(variable))
            if wavelet >=2:
                wavelet_x[:,i + 1] = pywt.upcoef('d', D2, wave, level=2, take=len(variable))
            if wavelet == 3:
                wavelet_x[:,i + 2] = pywt.upcoef('d', D1, wave, level=1, take=len(variable))
        x_arr = wavelet_x
    
    


    X = torch.from_numpy(np.array([x_arr[i:length + i,:] for i in range(len(x_arr) - length -1)]))
    Y = torch.from_numpy(np.array([[y_arr[length + i+1]] for i in range(len(y_arr) - length -1)]))
    return X,Y


def get_dataset(path_list : list,y : str ,x : list , length : int, method = "seq2one",normalize = "zscore", wavelet = 3, date = "20150131",test = None):
    """ Return a TensorDataset of all time series with a sliding window approach of given length"""
    X = None
    Y = None
    for path in path_list : 
        if method == "seq2one":
            df = filter_by_date(path,date = date, test = test, length = length)
            Xpath,Ypath = get_tensor_seq2one(df,y,x,length,normalize = normalize, wavelet = wavelet)
        else : 
            Xpath,Ypath = get_tensor(path,y,x,length,normalize = normalize, wavelet = wavelet)
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


def filter_by_date(path : str, date : str, test: bool = False, length: int = 0):
    """ Returns all predictions before a given date """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

    cutoff_date = pd.to_datetime(date, format="%Y%m%d")
    if not test :
        filtered = df[df["date"] < cutoff_date]
        
    elif test:
        start_date = cutoff_date  - pd.DateOffset(months=length)
        filtered = df[df["date"] < cutoff_date and df["date"] >= start_date]
    return filtered


if __name__ == "__main__":
    get_dataset(PATHS,Y_pred,X_COL,30)