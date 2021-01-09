import torch
from torch import nn
import numpy as np

'''
https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py
'''

class DeepSVDD(nn.Module):
    def __init__(
        self, 
        objective: str = "soft_boundary",
        R: float = 0.0,
        c = None,
        nu: float = 0.01,
        boundary_warm_up: int = 10,
        useGPU = True
    ):
        super(DeepSVDD, self).__init__()
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available(): self.device = torch.device("cuda")
        
        assert objective in ("one_class", "soft_boundary"), "Objective -> one_class / soft_boundary"
        self.objective = objective
        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device, requires_grad = False)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        assert self.nu<=1.0 and self.nu >0, "Condition for nu: 0 < nu <= 1"
        self.boundary_warm_up = boundary_warm_up
        
    def set_trainer(
        self,
        model,
        embeddings,
        lr: float = 1e-6,
        weight_decay = 1e-7,
        lr_scheduler_kwargs = {
            "factor": 0.5,
            "patience": 4,
            "threshold": 1e-4,
            "min_lr": 0,
            "eps": 1e-08,
            "verbose": True,
        }
    ):
        # Initialize hypersphere center c (if c not loaded)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **lr_scheduler_kwargs)
        if self.c is None: self.init_center(embeddings.flatten(start_dim = 1, end_dim = -1))
        print("INFO: SVDD Initialized")
        self.history = {
            "epoch_train_loss": list(),
            "epoch_val_loss": list(),
            "train_loss": list(),
            "val_loss": list()
        }
    
    def init_center(
        self,
        embeddings,
        eps: float = 1e-2
    ):
        mean_c = embeddings.detach().mean(dim = 0, keepdim = True)
        
#         # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
#         mean_c[(abs(mean_c) < eps) & (mean_c < 0)] = -eps
#         mean_c[(abs(mean_c) < eps) & (mean_c > 0)] = eps
        
        self.c = mean_c
        self.c = self.c.to(self.device)
        print("INFO: SVDD C Initialized", self.c.shape)
            
    def get_distance(
        self,
        embeddings
    ):
        return torch.sum((embeddings.flatten(start_dim = 1, end_dim = -1) - self.c)**2, dim = 1)
    
    def loss_function(
        self,
        embeddings
    ):
        dist = self.get_distance(embeddings)
        if self.objective == "soft_boundary":
            scores = dist - self.R ** 2
            loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        return loss, dist
    
    def get_radius(
        self,
        dist,
        nu
    ):
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    
    def update_radius(
        self,
        dist
    ):
        # Update hypersphere radius R on mini-batch distances
        if self.objective == 'soft_boundary':
            self.R.data = torch.tensor(self.get_radius(dist, self.nu), device=self.R.device)
    
    def get_scores(
        self,
        embeddings,
    ):
        scores = torch.sum((embeddings - self.c) ** 2, dim=1) # dist
        if self.objective == 'soft_boundary':
            scores = scores - self.R ** 2
        return scores
    
    def train_step(
        self,
        embeddings,
        updateR = True
    ):
        self.optimizer.zero_grad()
        loss, dist = self.loss_function(embeddings)
        loss.backward() # retain_graph=True
        self.optimizer.step()
        if updateR: self.update_radius(dist)
        return loss
    
    def val_step(
        self,
        embeddings
    ):
        with torch.no_grad():
            loss, dist = self.loss_function(embeddings)
        return loss