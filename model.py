import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers=nn.ModuleList()

        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i],layers[i+1]))
            nn.init.xavier_normal_(self.layers[i].weight)
            #nn.init.zores_(self.layers[i].bias)

    def forward(self,x,t):
        u=torch.cat((x,t),dim=1)
        for layer in self.layers:
            u=torch.tanh_(layer(u))
        return u
