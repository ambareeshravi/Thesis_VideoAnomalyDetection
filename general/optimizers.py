from .all_imports import torch

def ADAM(model, lr = 1e-3, betas=(0.9, 0.999), weight_decay = 1e-5):
    return torch.optim.Adam(model.parameters(), lr = lr, betas = betas, weight_decay = weight_decay)

def SGD(model, lr = 1e-3, momentum = 0.9):
    return torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)

def ADAGRAD(model, lr = 1e-2, lr_decay = 0, weight_decay = 1e-5):
    return torch.optim.Adagrad(model.parameters(), lr = lr, lr_decay = lr_decay, weight_decay = weight_decay)
    
def RMSProp(model, lr = 1e-2, weight_decay = 1e-5):
    return torch.optim.RMSprop(model.parameters(), lr = lr, weight_decay = weight_decay)

select_optimizer = {
    "adam": ADAM,
    "sgd": SGD,
    "adagrad": ADAGRAD,
    "rmsprop": RMSProp
}