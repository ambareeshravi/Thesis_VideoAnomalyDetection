from .all_imports import *

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim
    
class MSE_LOSS:
    def __init__(self, reduction = "mean"):
        self.loss = nn.MSELoss(reduction = reduction)
    
    def __call__(self, original, reconstructions):
        return self.loss(original, reconstructions)

class BCE_LOSS:
    def __init__(self, reduction = "mean"):
        self.loss = nn.BCELoss(reduction = reduction)
    
    def __call__(self, original, reconstructions):
        return self.loss(reconstructions, original)

class CONTRACTIVE_LOSS:
    def __init__(self, primary_loss = "mse", lamda = 1e-3):
        self.main_loss = MSE_LOSS(reduction = "mean")
        if "bce" in primary_loss: self.main_loss = BCE_LOSS(reduction = "mean")
        self.lamda = lamda
    
    def __call__(self, original, reconstructions, encodings, W):
        main_loss = self.main_loss(original, reconstructions)
#         W = torch.flatten(W, start_dim = 1, end_dim = -1)
#         dh = (encodings * (1 - encodings)).to(Config.device)
#         dh = torch.squeeze(torch.squeeze(dh, dim = -1), dim = -1)
#         contractive_loss = self.lamda * torch.sum(dh**2 * torch.sum(W**2, dim=-1)).to(Config.device)
        contractive_loss = (self.lamda * torch.norm(encodings*(1-encodings)) * torch.norm(W).to(Config.device))
        return main_loss + contractive_loss, {"MSE": main_loss, "CE": contractive_loss}
    
class PSNR_LOSS:
    def __init__(self, limit = 100):
        self.limit = limit
    
    def __call__(self, original, reconstructions):
        mse = torch.mean((original - reconstructions)**2)
        if mse == 0: psnr = self.limit
        else: psnr =  (10 * torch.log10(1. / mse)) * (self.limit / 100)
        return self.limit - psnr
    
class VARIATIONAL_LOSS:
    def __init__(self):
        self.bce_loss = BCE_LOSS(reduction = "mean")
        
    def __call__(self, original, reconstructions, mu, logvar):
        BCE = self.bce_loss(original, reconstructions)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, {"BCE": BCE, "KLD": KLD}
    
class WEIGHTED_SIMILARITY:
    def __init__(self, primary_loss = "mse", weights = [1.0, 1.0], asImages = True):
        self.main_loss = MSE_LOSS(reduction = "mean")
        if "bce" in primary_loss: self.main_loss = BCE_LOSS(reduction = "mean")
        self.ssim_loss = SSIM(data_range = 1.0, nonnegative_ssim=True)
        self.weights = weights
        self.asImages = asImages
        
    def transform(self, x):
        # Gray Image to 3 channels in images
        if self.asImages and x.shape[-3] == 1: x = x.repeat(1, 3, 1, 1)
        # Gray frames to 3 channels in video
        if not self.asImages and x.shape[-4] == 1: x = x.repeat(1, 3, 1, 1, 1)
        # BatchSize x channels x frames x h x w
        if self.asImages: return x
        # transpose n_frame and channels for video
        else: return x.transpose(1,2).flatten(start_dim = 0, end_dim = 1)
    
    def __call__(self, original, reconstructions):
        return (self.weights[0] * self.main_loss(original, reconstructions)) + (self.weights[1] * 100 * (1 - self.ssim_loss.forward(self.transform(original), self.transform(reconstructions))))
    
class QUALITY_LOSS(WEIGHTED_SIMILARITY):
    def __init__(self):
        WEIGHTED_SIMILARITY.__init__(self)
        self.psnr = PSNR_LOSS(limit = 1)
        
    def __call__(self, original, reconstructions):
        return self.psnr(original, reconstructions) + (1 - self.ssim_loss.forward(self.transform(original), self.transform(reconstructions)))
            
class MahalanobisLayer(nn.Module):
    def __init__(self, dim = 300):
        super(MahalanobisLayer, self).__init__()
        self.register_buffer('S', torch.eye(dim))
        self.register_buffer('S_inv', torch.eye(dim))

    def cov(self, x):
        x -= torch.mean(x, dim=0)
        return 1 / (x.size(0) - 1) * x.t().mm(x)

    def forward(self, x):
        delta = x - torch.mean(x, dim = 0)
        self.S_inv = torch.pinverse(self.cov(delta + 1e-10))
        m = torch.mm(torch.mm(delta, self.S_inv), delta.t())
        return torch.mean(torch.diag(m))

class MANIFOLD_LOSS:
    def __init__(self, primary_loss = "mse", weights = [1.0, 1.0]):
        self.main_loss = MSE_LOSS(reduction = "mean")
        if "bce" in primary_loss: self.main_loss = BCE_LOSS(reduction = "mean")
        self.mahalanobis_loss = MahalanobisLayer()
        self.weights = weights
    
    def __call__(self, original, reconstruction, encoding):
        return (self.weights[0] * self.main_loss(original, reconstruction)) + (self.weights[1] * self.mahalanobis_loss(encoding))
    
def max_norm(w, max_clip):
    norms = torch.norm_except_dim(w, dim=0).flatten()
    desired = torch.clip(norms, 0, max_clip)
    w *= (desired / (1e-10 + norms))
    return w
    
select_loss = {
    "mse": MSE_LOSS(),
    "bce": BCE_LOSS(),
    "psnr": PSNR_LOSS(),
    "weighted": WEIGHTED_SIMILARITY(),
    "quality": QUALITY_LOSS(),
    "manifold": MANIFOLD_LOSS()
}