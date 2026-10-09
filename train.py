import torch
from model import PINN
from physics import physics_loss

x_train=torch.linspace(0,1,100,requires_grad=True).view(-1,1)
t_train=torch.linspace(0,1,100,requires_grad=True).view(-1,1)

model=PINN([2,50,50,1])
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
alpha=0.001

for epoch in range(10000):
    optimizer.zero_grad()
    loss=physics_loss(model,x_train,t_train,alpha)
    loss.backward()
    optimizer.step()
    if epoch%1000==0:
        print(f'Epoch{epoch},Loss:{loss.item():.6f}')