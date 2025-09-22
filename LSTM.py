import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import base
import matplotlib.pyplot as plt

class LSTM_M2M(nn.Module):
    """
    Many2Many LSTM. Computes Y(t) based on X(t) and X(t'<t) : Output Y(0 < t < T)
    """
    def __init__(self,input_size,hidden_size,num_layers,output_size,dropout = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size,output_size)


    def forward(self,x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        lstm_out,_ = self.lstm(x,(h0,c0))

        lstm_out.float()
        batch_size, seq_len, hidden_size = lstm_out.size()
        output = self.fc(lstm_out)
        return output
     


class LSTM_seq2one(nn.Module):

    """
    Predicts Y(t+1) based on X(t) and X(t'< t) : output Y(T+1)
    """

    def __init__(self,input_size,hidden_size,num_layers,output_size,dropout = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size,output_size)


    def forward(self,x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        _ ,out = self.lstm(x,(h0,c0))
        lstm_out = out[0][-1]

        # print(f"e 0 : {lstm_out.shape}")
        # print(f"x : {x.shape}")
        # print(f"out : {lstm_out.shape}")

        lstm_out.float()
        output = self.fc(lstm_out)
        return output
    


def train(model,dataloader,epochs,criterion,optimizer,device):

    model.train()
    model.to(device)
    tot_loss = np.zeros(epochs)
    for epoch in range(epochs):
        print(f"epoch : {epoch}")
        for data,label in dataloader:
            optimizer.zero_grad()
            data = torch.nan_to_num(data)
            data = data.type(torch.float32)
            # print( torch.isnan(data).any())
            label = label.type(torch.float32)
            output = model(data)
            #print(output)

            #print(output.shape)
          
            #print(label.shape) 
            loss = criterion(output,label) 
            tot_loss[epoch] += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print(tot_loss[epoch])
    return tot_loss


if __name__ == "__main__":
    data = base.get_dataset(base.PATHS,base.Y_pred,base.X_COL,30)
    dataloader = torch.utils.data.dataloader.DataLoader(data,batch_size = 5,shuffle = True)
    model = LSTM_seq2one(20,32,2,1)
    optimizer = optim.Adam(model.parameters(),lr = 0.003)
    criterion = nn.MSELoss()
    loss = train(model,dataloader,100,criterion,optimizer,"cpu")

    plt.plot(loss)
    plt.ylabel("MSE")
    plt.xlabel("EPOCHS")
    plt.title("Loss per Epoch on Reduced Dataset")
    #plt.text("Model : LSTM; Optimizer : Adam; Criterion : MSELoss; Learning rate = 0.001")
    plt.savefig("LSTM_training_result_lr0.003.png")
    plt.close()