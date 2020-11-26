import sys
sys.path.append("..")
from general import *

from general.model_utils import *

class PatchWise_C2D(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,128,256,128]
    ):
        super(PatchWise_C2D, self).__init__()
        self.channels = channels
        self.filters_count = filters_count

        self.encoder = nn.Sequential(
            C2D_ACB(self.channels, self.filters_count[0], 5, 2),
            C2D_ACB(self.filters_count[0], self.filters_count[1], 5, 2),
            C2D_ACB(self.filters_count[1], self.filters_count[2], 5, 2),
            C2D_ACB(self.filters_count[2], self.filters_count[3], 3, 2),
            C2D_ACB(self.filters_count[3], self.filters_count[4], 3, 2),
        )
        
        self.decoder = nn.Sequential(
            CT2D_ADB(self.filters_count[4], self.filters_count[3], 3, 2),
            CT2D_ADB(self.filters_count[3], self.filters_count[2], 5, 2),
            CT2D_ADB(self.filters_count[2], self.filters_count[1], 5, 2),
            CT2D_ADB(self.filters_count[1], self.filters_count[0], 5, 2),
            CT2D_ADB(self.filters_count[0], self.channels, 4, 1),
        )
    
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class PatchWise_C3D(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,128,256,128]
    ):
        super(PatchWise_C3D, self).__init__()
        self.channels = channels
        self.filters_count = filters_count

        self.encoder = nn.Sequential(
            C3D_BN_A(self.channels, self.filters_count[0], 3, 2),
            C3D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2),
            C3D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2),
            C3D_BN_A(self.filters_count[2], self.filters_count[3], (1,3,3), (1,2,2)),
        )
        
        self.decoder = nn.Sequential(
            CT3D_BN_A(self.filters_count[3], self.filters_count[2], (1,3,3), (1,2,2)),
            CT3D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            CT3D_BN_A(self.filters_count[1], self.filters_count[0], 3, 2),
            CT3D_BN_A(self.filters_count[0], self.channels, 4, 2),
        )
    
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings