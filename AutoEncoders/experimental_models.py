from C2D_Models import *

class C2D_AE_ANYRES(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        encoder_activation = "tanh",
        conv_type = "conv2d"
    ):
        super(C2D_AE_ANYRES, self).__init__()
        self.__name__ = "C2D_AE_ANYRES"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[4],4,4]
        
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 5, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 5, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 5, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 5, 2, conv_type = conv_type, activation_type = encoder_activation),
        )
        
        self.decoder = nn.Sequential(
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 5, 2),
            CT2D_BN_A(self.filters_count[3], self.filters_count[2], 5, 2),
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 6, 2),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 5, 2),
            CT2D_BN_A(self.filters_count[0], self.channels, 4, 2, activation_type = "sigmoid"),
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class C2D_AE_128_POOL(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [128,256,512],
        encoder_activation = "tanh",
    ):
        super(C2D_AE_128_POOL, self).__init__()
        self.__name__ = "C2D_3x3_POOL"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[2],4,4]
        
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2),
            nn.MaxPool2d((2,2)),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2),
            nn.MaxPool2d((2,2)),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 4, 1, activation_type="tanh"),
        )
        
        self.decoder = nn.Sequential(
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 4, 1),
            nn.Upsample(scale_factor=2),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 5, 2),
            nn.Upsample(scale_factor=2),
            CT2D_BN_A(self.filters_count[0], self.channels, 6, 2, activation_type="sigmoid"),
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings