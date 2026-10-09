import torch.autograd

def physics_loss(model,x,t,alpha):
    u=model(x,t)
    u_t=torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_x=torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    residual=u_t-alpha*u_xx

    return torch.mean(residual**2)
