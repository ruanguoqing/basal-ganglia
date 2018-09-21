import torch


def torchify(y):
    return torch.tensor(y).type(torch.FloatTensor)


def make_indicator(y_tensor, n_dim=None):
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dim = n_dim if n_dim is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dim).scatter_(1, y_tensor, 1)
    return y_one_hot
