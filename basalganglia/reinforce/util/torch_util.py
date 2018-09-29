import numpy as np
import torch, torch.autograd as A
from gym.spaces import Box, Discrete


def torchify(y, type=Box):
    if type is Box:
        return torch.Tensor(y).type(torch.FloatTensor)
    elif type is Discrete:
        return torch.Tensor(y).type(torch.LongTensor)


def make_indicator(y_tensor, n_dim=None):
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dim = n_dim if n_dim is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dim).scatter_(1, y_tensor, 1)
    return y_one_hot


def hvp(y, x, v):
    grad = A.grad(y, x(), create_graph=True, retain_graph=True)
    flat_grad = make_flat_from(grad)
    grad_v = (flat_grad * v).sum()
    grad_grad = A.grad(grad_v, x(), retain_graph=True)
    flat_grad_grad = make_flat_from(grad_grad).data
    return flat_grad_grad + 0.1 * v


def do_conjugate_gradient(f_Ax, b, n_iters=10, tolerance=1e-10):
    p = b.clone()
    r = b.clone()
    x = torch.zeros(b.size())
    for i in range(n_iters):
        residue = (r * r).sum()
        Ap = f_Ax(p)
        alpha = residue / ((p * Ap).sum() + 1e-8)
        x += alpha * p
        r -= alpha * Ap

        new_residue = (r * r).sum()
        if new_residue < tolerance:
            break

        beta = new_residue / (residue + 1e-8)
        p = r + beta * p
    return x


def make_flat_from(v):
    return torch.cat([g.contiguous().view(-1) for g in v])


def make_flat_from_model(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_model_from_flat(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def set_grad_from_flat(model, flat_grad):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.grad.data.copy_(flat_grad[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
