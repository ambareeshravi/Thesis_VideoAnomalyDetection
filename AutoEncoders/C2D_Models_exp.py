import sys
sys.path.append("..")
from general import *
from general.model_utils import *

class C2D128_1(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [1024,512,512,256,256,256],
        conv_type = "conv2d"
    ):
        super(C2D128_1, self).__init__()
        self.channels = channels
        self.filters_count = filters_count
        
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 3, 2, conv_type = conv_type),
            nn.Conv2d(self.filters_count[4], self.filters_count[5], 3, 1),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            CT2D_BN_A(self.filters_count[5], self.filters_count[4], 3, 1),
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 3, 2),
            CT2D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2),
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 3, 2),
            CT2D_BN_A(self.filters_count[0], self.channels, 4, 2, activation_type = "sigmoid"),
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class C2D128_2(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [1024,512,512,256,256,128],
        conv_type = "conv2d"
    ):
        super(C2D128_2, self).__init__()
        self.channels = channels
        self.filters_count = filters_count
        
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 3, 2, conv_type = conv_type),
            nn.Conv2d(self.filters_count[4], self.filters_count[5], 3, 1),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            CT2D_BN_A(self.filters_count[5], self.filters_count[4], 3, 1),
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 3, 2),
            CT2D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2),
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 3, 2),
            CT2D_BN_A(self.filters_count[0], self.channels, 4, 2, activation_type = "sigmoid"),
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings