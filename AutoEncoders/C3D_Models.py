import sys
sys.path.append("..")
from general import *
from general.model_utils import *

class Generic_C3D_AE(nn.Module):
    def __init__(self,
                 encoder_layer_info = [
                        [64,3,1,0,1],
                        [64,3,2,0,1],
                        [64,3,2,0,1],
                        [64,(2,3,3),(1,2,2),0,1],
                        [128,(1,3,3),(1,2,2),0,1],
                        [128,(1,3,3),(1,2,2),0,1]
                    ],
                    decoder_layer_info=[
                        [128,3,2,0,0],
                        [128,(3,5,5),2,0,0],
                        [64,(3,5,5),2,0,0],
                        [64,(2,5,5),(1,2,2),0,0],
                        [64,(1,5,5),(1,2,2),(0,1,1),0],
                        [64,(1,5,5),1,0,0],
                        [64,(1,2,2),1,0,0],
                    ],
                 channels = 3,
                 image_size = 128,
                 n_frames = 16,
                 debug = False):
        super(Generic_C3D_AE, self).__init__()
        self.__name__ = "C3D_Generic"
        self.channels = channels
        self.debug = debug
        
        encoder_layers = list()
        in_channels = self.channels
        
        for idx, (out_channels, kernel_size, stride, padding, dilation) in enumerate(encoder_layer_info):
            encoder_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding = padding, dilation = dilation))
            if idx == len(encoder_layer_info) - 1:
                encoder_layers.append(BN_A(out_channels))
            else:
                encoder_layers.append(nn.BatchNorm3d(out_channels))
                encoder_layers.append(nn.Tanh())
            in_channels = out_channels
            
        self.encoder = nn.Sequential(*encoder_layers)
        if self.debug: print("Encoding dim: ", self.encoder(torch.rand(1,self.channels,n_frames,image_size,image_size)).shape)
        
        decoder_layers = list()
        in_channels = out_channels
        
        for idx, (out_channels, kernel_size, stride, padding, output_padding) in enumerate(decoder_layer_info):
            if idx < len(decoder_layer_info) - 1:
                decoder_layers.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding = padding, output_padding = output_padding))
                decoder_layers.append(BN_A(out_channels))
                in_channels = out_channels
            else:
                decoder_layers.append(nn.ConvTranspose3d(in_channels, self.channels, kernel_size, stride, padding = padding, output_padding = output_padding))
                decoder_layers.append(nn.Sigmoid())
            
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings

class C3D_AE_3x3(nn.Module):
    def __init__(
        self,
        channels = 3,
        filter_count = [64,64,64,96,96,128],
        conv_type = "conv3d"
    ):
        super(C3D_AE_3x3, self).__init__()
        self.__name__ = "C3D_3x3_128"
        self.channels = channels
        self.filter_count = filter_count
        
        self.encoder = nn.Sequential(
            C3D_BN_A(self.channels, self.filter_count[0], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filter_count[0], self.filter_count[1], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filter_count[1], self.filter_count[2], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filter_count[2], self.filter_count[3], (1,3,3), (1,2,2), conv_type = conv_type),
            C3D_BN_A(self.filter_count[3], self.filter_count[4], (1,4,4), (1,1,1), conv_type = conv_type, activation_type="tanh"),
#             C3D_BN_A(self.filter_count[4], self.filter_count[5], (1,2,2), (1,1,1), conv_type = conv_type),
        )
        
        self.decoder = nn.Sequential(
#             CT3D_BN_A(self.filter_count[5], self.filter_count[4], (1,2,2), (1,1,1)),
            CT3D_BN_A(self.filter_count[4], self.filter_count[3], (1,4,4), (1,1,1)),
            CT3D_BN_A(self.filter_count[3], self.filter_count[2], (1,3,3), (1,2,2)),
            CT3D_BN_A(self.filter_count[2], self.filter_count[1], 3, 2),
            CT3D_BN_A(self.filter_count[1], self.filter_count[0], 4, 2),
            CT3D_BN_A(self.filter_count[0], self.filter_count[0], 4, 2),
            C3D_BN_A(self.filter_count[0], self.channels, 3, 1, activation_type="sigmoid")
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class C3D_AE_Res_3x3(nn.Module):
    def __init__(
        self,
        channels = 3,
        filter_count = [64,64,64,96,96,128],
        conv_type = "conv3d"
    ):
        super(C3D_AE_Res_3x3, self).__init__()
        self.__name__ = "C3D_3x3_128_RES"
        self.channels = channels
        self.filter_count = filter_count
        
        self.encoder = nn.Sequential(
            C3D_BN_A(self.channels, self.filter_count[0], 3, 2, conv_type = conv_type),
            C3D_Res(self.filter_count[0], 3),
            C3D_BN_A(self.filter_count[0], self.filter_count[1], 3, 2, conv_type = conv_type),
            C3D_Res(self.filter_count[1], 3),
            C3D_BN_A(self.filter_count[1], self.filter_count[2], 3, 2, conv_type = conv_type),
            C3D_Res(self.filter_count[2], 3),
            C3D_BN_A(self.filter_count[2], self.filter_count[3], (1,3,3), (1,2,2), conv_type = conv_type),
            C3D_Res(self.filter_count[3], 3),
            C3D_BN_A(self.filter_count[3], self.filter_count[4], (1,4,4), (1,1,1), conv_type = conv_type, activation_type="tanh"),
#             C3D_Res(self.filter_count[4], 3),
#             C3D_BN_A(self.filter_count[4], self.filter_count[5], (1,2,2), (1,1,1), conv_type = conv_type, activation_type="tanh")
        )
        
        self.decoder = nn.Sequential(
#             CT3D_BN_A(self.filter_count[5], self.filter_count[4], (1,2,2), (1,1,1)),
#             CT3D_Res(self.filter_count[4], 3),
            CT3D_BN_A(self.filter_count[4], self.filter_count[3], (1,4,4), (1,1,1)),
            CT3D_Res(self.filter_count[3], 3),
            CT3D_BN_A(self.filter_count[3], self.filter_count[2], (1,3,3), (1,2,2)),
            CT3D_Res(self.filter_count[2], 3),
            CT3D_BN_A(self.filter_count[2], self.filter_count[1], 3, 2),
            CT3D_Res(self.filter_count[1], 3),
            CT3D_BN_A(self.filter_count[1], self.filter_count[0], 4, 2),
            CT3D_Res(self.filter_count[0], 3),
            CT3D_BN_A(self.filter_count[0], self.filter_count[0], 4, 2),
            C3D_BN_A(self.filter_count[0], self.channels, 3, 1, activation_type = "sigmoid"),
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings

class C3D2D_AE_3x3(nn.Module):
    def __init__(
        self,
        channels = 3,
        filter_count = [64,64,64,96,96,128],
        conv_type = "conv3d"
    ):
        super(C3D2D_AE_3x3, self).__init__()
        self.__name__ = "C3D2D_3x3_128"
        self.channels = channels
        self.filter_count = filter_count
        
        self.encoder = nn.Sequential(
            C3D_BN_A(self.channels, self.filter_count[0], 3, 1, conv_type = conv_type),
            C3D_BN_A(self.filter_count[0], self.filter_count[1], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filter_count[1], self.filter_count[2], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filter_count[2], self.filter_count[3], (2,5,5), (1,3,3), conv_type = conv_type),
            nn.Flatten(start_dim=1, end_dim=2),
            C2D_BN_A(self.filter_count[3], self.filter_count[4], 5, 2, activation_type = "tanh"),
        )
        
        self.decoder1 = nn.Sequential(
#             CT2D_BN_A(self.filter_count[5], self.filter_count[4], 2, 1),
            CT2D_BN_A(self.filter_count[4], self.filter_count[3], 3, 2),
        )
        
        self.decoder2 = nn.Sequential(
            CT3D_BN_A(self.filter_count[3], self.filter_count[2], 3, 2),
            CT3D_BN_A(self.filter_count[2], self.filter_count[1], 3, 2),
            CT3D_BN_A(self.filter_count[1], self.filter_count[0], 4, 2),
            CT3D_BN_A(self.filter_count[0], self.filter_count[0], 4, 2),
            C3D_BN_A(self.filter_count[0], self.channels, 3, (2,1,1), activation_type = "sigmoid")
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        prilim_out = self.decoder1(encodings)
        prilim_out = prilim_out.unsqueeze(dim=2)
        reconstructions = self.decoder2(prilim_out)
        return reconstructions, encodings
    
C3D_MODELS_DICT = {
    128: {
        "generic": Generic_C3D_AE,
        "vanilla": {
            "3x3": C3D_AE_3x3
        },
        "res": {
            "3x3": C3D_AE_Res_3x3
        },
        "3D2D": C3D2D_AE_3x3
    }
}