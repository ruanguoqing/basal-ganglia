import numpy as np
import torch, torch.autograd as A

def grad(f, x):
    return A.grad(f, x, create_graph=True, retain_graph=True)[0].detach().numpy()

def hess(f, x, y):
    g = A.grad(f, x, create_graph=True, retain_graph=True)
    return np.array(list(A.grad(g[0][i], y, create_graph=True, retain_graph=True)[0].detach().numpy()
                         for i in range(g[0].shape[0])))
