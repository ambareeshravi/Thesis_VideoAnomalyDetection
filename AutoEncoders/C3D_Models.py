import sys
sys.path.append("..")
from general import *
from general.model_utils import *

class C3D_AE_Generic(nn.Module):
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
        super(C3D_AE_Generic, self).__init__()
        self.__name__ = "C3D_AE_Generic_Generic|"
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

class C3D_AE_128_3x3(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv3d"
    ):
        super(C3D_AE_128_3x3, self).__init__()
        self.__name__ = "C3D_AE_128_3x3|"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1, self.filters_count[4], 1, 4, 4]
        
        self.encoder = nn.Sequential(
            C3D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filters_count[2], self.filters_count[3], (1,3,3), (1,2,2), conv_type = conv_type),
            C3D_BN_A(self.filters_count[3], self.filters_count[4], (1,4,4), (1,1,1), conv_type = conv_type, activation_type="tanh"),
#             C3D_BN_A(self.filters_count[4], self.filters_count[5], (1,2,2), (1,1,1), conv_type = conv_type),
        )
        
        self.decoder = nn.Sequential(
#             CT3D_BN_A(self.filters_count[5], self.filters_count[4], (1,2,2), (1,1,1)),
            CT3D_BN_A(self.filters_count[4], self.filters_count[3], (1,4,4), (1,1,1)),
            CT3D_BN_A(self.filters_count[3], self.filters_count[2], (1,3,3), (1,2,2)),
            CT3D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            CT3D_BN_A(self.filters_count[1], self.filters_count[0], 4, 2),
            CT3D_BN_A(self.filters_count[0], self.filters_count[0], 4, 2),
            C3D_BN_A(self.filters_count[0], self.channels, 3, 1, activation_type="sigmoid")
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class C3D_AE_128_3x3_Res(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv3d"
    ):
        super(C3D_AE_128_3x3_Res, self).__init__()
        self.__name__ = "C3D_AE_128_3x3_RES|"
        self.channels = channels
        self.filters_count = filters_count
        
        self.encoder = nn.Sequential(
            C3D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C3D_Res(self.filters_count[0], 3),
            C3D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C3D_Res(self.filters_count[1], 3),
            C3D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C3D_Res(self.filters_count[2], 3),
            C3D_BN_A(self.filters_count[2], self.filters_count[3], (1,3,3), (1,2,2), conv_type = conv_type),
            C3D_Res(self.filters_count[3], 3),
            C3D_BN_A(self.filters_count[3], self.filters_count[4], (1,4,4), (1,1,1), conv_type = conv_type, activation_type="tanh"),
#             C3D_Res(self.filters_count[4], 3),
#             C3D_BN_A(self.filters_count[4], self.filters_count[5], (1,2,2), (1,1,1), conv_type = conv_type, activation_type="tanh")
        )
        
        self.decoder = nn.Sequential(
#             CT3D_BN_A(self.filters_count[5], self.filters_count[4], (1,2,2), (1,1,1)),
#             CT3D_Res(self.filters_count[4], 3),
            CT3D_BN_A(self.filters_count[4], self.filters_count[3], (1,4,4), (1,1,1)),
            CT3D_Res(self.filters_count[3], 3),
            CT3D_BN_A(self.filters_count[3], self.filters_count[2], (1,3,3), (1,2,2)),
            CT3D_Res(self.filters_count[2], 3),
            CT3D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            CT3D_Res(self.filters_count[1], 3),
            CT3D_BN_A(self.filters_count[1], self.filters_count[0], 4, 2),
            CT3D_Res(self.filters_count[0], 3),
            CT3D_BN_A(self.filters_count[0], self.filters_count[0], 4, 2),
            C3D_BN_A(self.filters_count[0], self.channels, 3, 1, activation_type = "sigmoid"),
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings

class C3D2D_AE_128_3x3(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv3d"
    ):
        super(C3D2D_AE_128_3x3, self).__init__()
        self.__name__ = "C3D2D_AE_128_3x3|"
        self.channels = channels
        self.filters_count = filters_count
        
        self.encoder = nn.Sequential(
            C3D_BN_A(self.channels, self.filters_count[0], 3, 1, conv_type = conv_type),
            C3D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filters_count[2], self.filters_count[3], (2,5,5), (1,3,3), conv_type = conv_type),
            nn.Flatten(start_dim=1, end_dim=2),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 5, 2, activation_type = "tanh"),
        )
        
        self.decoder1 = nn.Sequential(
#             CT2D_BN_A(self.filters_count[5], self.filters_count[4], 2, 1),
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 3, 2),
        )
        
        self.decoder2 = nn.Sequential(
            CT3D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2),
            CT3D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            CT3D_BN_A(self.filters_count[1], self.filters_count[0], 4, 2),
            CT3D_BN_A(self.filters_count[0], self.filters_count[0], 4, 2),
            C3D_BN_A(self.filters_count[0], self.channels, 3, (2,1,1), activation_type = "sigmoid")
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        prilim_out = self.decoder1(encodings)
        prilim_out = prilim_out.unsqueeze(dim=2)
        reconstructions = self.decoder2(prilim_out)
        return reconstructions, encodings

class C3D_AE_Multi_3x3(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv3d"
    ):
        super(C3D_AE_Multi_3x3, self).__init__()
        self.__name__ = "C3D_AE_MULTI_3x3|"
        self.channels = channels
        self.filters_count = filters_count
        
        self.encoder = nn.Sequential(
            C3D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C3D_BN_A(self.filters_count[2], self.filters_count[3], (1,3,3), (1,2,2), conv_type = conv_type),
            C3D_BN_A(self.filters_count[3], self.filters_count[4], (1,3,3), (1,1,1), conv_type = conv_type, activation_type="tanh"),
#             C3D_BN_A(self.filters_count[4], self.filters_count[5], (1,2,2), (1,1,1), conv_type = conv_type),
        )
        
        self.decoder = nn.Sequential(
#             CT3D_BN_A(self.filters_count[5], self.filters_count[4], (1,2,2), (1,1,1)),
            CT3D_BN_A(self.filters_count[4], self.filters_count[3], (1,3,3), (1,1,1)),
            CT3D_BN_A(self.filters_count[3], self.filters_count[2], (1,3,3), (1,2,2)),
            CT3D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            CT3D_BN_A(self.filters_count[1], self.filters_count[0], 3, 2),
            CT3D_BN_A(self.filters_count[0], self.channels, 4, 2, activation_type="sigmoid")
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings

class C3D_AE_128_3x3_VAE(C3D_AE_128_3x3):
    def __init__(
        self,
        isTrain = True,
        channels = 3,
        filters_count = [64,64,64,96,128,128],
        conv_type = "conv3d"
    ):
        C3D_AE_128_3x3.__init__(self, channels = channels, filters_count = filters_count, conv_type = conv_type)
        self.__name__ = "C3D_AE_128_3x3_VAE|"
        self.view_shape = tuple([-1] + self.embedding_dim[1:])
        self.embedding_dim = np.product(self.embedding_dim)
        self.isTrain = isTrain
        
        self.fc_mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
                
    def latent_sample(self, mu, logvar):
        if self.isTrain:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def vae_loss(self, original, reconstruction, mu, logvar, variational_beta = 1.0):
        recon_loss = F.binary_cross_entropy(reconstruction.flatten(), original.flatten(), reduction='mean')
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + (variational_beta * kldivergence)

    def forward(self, x):
        # Encoder
        encodings = self.encoder(x)
        mu = self.fc_mu(encodings)
        logvar = self.fc_logvar(encodings)
        
        latent = self.latent_sample(mu, logvar)
        reconstructions = self.decoder(latent.reshape(*self.view_shape))
        return reconstructions, mu, logvar
    
C3D_MODELS_DICT = {
    128: {
        "generic": C3D_AE_Generic,
        "vanilla": {
            "3x3": C3D_AE_128_3x3
        },
        "vae": {
            "3x3": C3D_AE_128_3x3_VAE
        },
        "res": {
            "3x3": C3D_AE_128_3x3_Res
        },
        "3D2D": C3D2D_AE_128_3x3,
        "multi_resolution": C3D_AE_Multi_3x3
    }
}

class C23D(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,96,96,128]
    ):
        super(C23D, self).__init__()
        self.__name__  = "C23D_AE|"
        self.channels = channels
        self.filters_count = filters_count
        
        self.c2d_encoder = nn.Sequential(
            TimeDistributed(C2D_BN_A(self.channels, self.filters_count[0], 3, 2)),
            TimeDistributed(C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2)),
            TimeDistributed(C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2)),
        )
        
        self.c3d_encoder = nn.Sequential(
            C3D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2),
            C3D_BN_A(self.filters_count[3], self.filters_count[4], 3, 2)
        )
        
        self.ct3d_decoder = nn.Sequential(
            CT3D_BN_A(self.filters_count[4], self.filters_count[3], 3, 2),
            CT3D_BN_A(self.filters_count[3], self.filters_count[2], (4,3,3), 2),
        )
        
        self.ct2d_decoder = nn.Sequential(
            TimeDistributed(CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2)),
            TimeDistributed(CT2D_BN_A(self.filters_count[1], self.filters_count[0], 3, 2)),
            TimeDistributed(CT2D_BN_A(self.filters_count[0], self.channels, 4, 2)),
        )
        
    def forward(self, x):
        c2d_out = self.c2d_encoder(x.permute(0,2,1,3,4))
        c3d_out = self.c3d_encoder(c2d_out.permute(0,2,1,3,4))
        ct3d_out = self.ct3d_decoder(c3d_out)
        ct2d_out = self.ct2d_decoder(ct3d_out.permute(0,2,1,3,4))
        encodings = c3d_out.permute(0,2,1,3,4)
        reconstructions = ct2d_out.permute(0,2,1,3,4)
        return reconstructions, encodings