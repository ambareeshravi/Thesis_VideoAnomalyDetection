import torch
from torch import nn

class FrameFeaturePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, isTrain = True, useGPU = True):
        super(FrameFeaturePredictor, self).__init__()
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available(): self.device = torch.device("cuda:0")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTM(self.input_dim, self.hidden_dim, batch_first = True)
        self.linear1 = nn.Linear(self.hidden_dim, self.input_dim)
        self.act1 = nn.LeakyReLU()
        self.isTrain = isTrain
        
    def zero_state(self, batch_size):
        return tuple([torch.zeros(1, batch_size, self.hidden_dim).to(self.device)]*2)
            
    def unroll(self, x, future_steps = 0):
        states = self.zero_state(1)
        
        outputs = list()
        for ts in range(len(x) + future_steps):
            try:
                input_vec = x[ts].unsqueeze(dim = 0)
            except: 
                input_vec = output
            output, states = self.forward(input_vec, states)
            outputs.append(output)
        return torch.cat(outputs, dim = 0)
    
    def forward(self, x, states):
        output, states = self.lstm1(x, states)
        output = self.linear1(output)
        output = self.act1(output)
        return output, states