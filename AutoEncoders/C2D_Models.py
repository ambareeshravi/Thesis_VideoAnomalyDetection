import sys
sys.path.append("..")

from torch.nn import functional as F

from general import *
from general.model_utils import *

class Generic_C2D_AE(nn.Module):
    def __init__(self,
                 encoder_layer_info=[
                    [32,5,3,0,1],
                    [32,3,2,0,1],
                    [32,3,2,0,1], 
                    [32,3,2,0,1]
                 ],
                 decoder_layer_info=[
                    [32,3,2,0,0],
                    [32,4,2,0,0],
                    [32,4,2,0,0],
                    [32,5,3,0,0]
                 ],
                 channels = 3,
                 image_size = 224
                ):
        super(Generic_C2D_AE, self).__init__()
        self.__name__ = "C2D_Generic"
        self.channels = channels
        
        encoder_layers = list()
        in_channels = self.channels
        output_shape = image_size
        
        for idx, (out_channels, kernel_size, stride, padding, dilation) in enumerate(encoder_layer_info):
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = padding, dilation = dilation))
            if idx == len(encoder_layer_info) - 1:
                encoder_layers.append(nn.BatchNorm2d(out_channels))
                encoder_layers.append(nn.Tanh())
            else:
                encoder_layers.append(BN_A(out_channels, is3d=False))
            in_channels = out_channels
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = list()
        in_channels = out_channels
        
        for idx, (out_channels, kernel_size, stride, padding, output_padding) in enumerate(decoder_layer_info):
            if idx < len(decoder_layer_info) - 1:
                decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding = padding, output_padding = output_padding))
                decoder_layers.append(BN_A(out_channels, is3d=False))
                in_channels = out_channels
            else:
                decoder_layers.append(nn.ConvTranspose2d(in_channels, self.channels, kernel_size, stride, padding = padding, output_padding = output_padding))
                decoder_layers.append(nn.Sigmoid())
            
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings

class C2D_AE_128_3x3(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,64,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_3x3, self).__init__()
        self.__name__ = "C2D_3x3_128"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,128,4,4]
        
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 4, 1, conv_type = conv_type, activation_type = "tanh"),
#             C2D_BN_A(self.filters_count[4], self.filters_count[5], 2, 1, conv_type = conv_type, activation_type = "tanh"),
        )
        
        self.decoder = nn.Sequential(
#             CT2D_BN_A(self.filters_count[5], self.filters_count[4], 2, 1),
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 4, 1),
            CT2D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2),
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 4, 2),
            CT2D_BN_A(self.filters_count[0], self.filters_count[0], 4, 2),
            C2D_BN_A(self.filters_count[0], self.channels, 3, 1, activation_type = "sigmoid")
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class C2D_AE_128_5x5(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,64,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_5x5, self).__init__()
        self.__name__ = "C2D_5x5_128"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,128,4,4]
        
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 5, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 5, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 5, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 4, 3, conv_type = conv_type, activation_type="tanh"),
#             C2D_BN_A(self.filters_count[3], self.filters_count[4], 3, 2, conv_type = conv_type, activation_type="tanh"),
        )
        
        self.decoder = nn.Sequential(
#             CT2D_BN_A(self.filters_count[4], self.filters_count[3], 3, 2),
            CT2D_BN_A(self.filters_count[3], self.filters_count[2], 4, 3),
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 5, 2),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 7, 2),
            CT2D_BN_A(self.filters_count[0], self.filters_count[0], 7, 2),
            C2D_BN_A(self.filters_count[0], self.channels, 4, 1, activation_type="sigmoid")
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings

class C2D_AE_128_3x3_VAE(C2D_AE_128_3x3):
    def __init__(
        self,
        isTrain = True,
        channels = 3,
        filters_count = [64,64,64,64,96,128],
        conv_type = "conv2d"
    ):
        C2D_AE_128_3x3.__init__(self, channels = channels, filters_count = filters_count, conv_type = conv_type)
        self.__name__ = "C2D_3x3_128_VAE"
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
        recon_loss = F.binary_cross_entropy(reconstruction.flatten(), original.flatten(), reduction='sum')
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
    
class C2D_AE_128_5x5_VAE(C2D_AE_128_5x5):
    def __init__(
        self,
        isTrain = True,
        channels = 3,
        filters_count = [64,64,64,64,96,128],
        conv_type = "conv2d"
    ):
        C2D_AE_128_5x5.__init__(self, channels = channels, filters_count = filters_count, conv_type = conv_type)
        self.__name__ = "C2D_5x5_128_VAE"
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
        recon_loss = F.binary_cross_entropy(reconstruction.flatten(), original.flatten(), reduction='sum')
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

class C2D_AE_3x3_Res(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,64,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_3x3_Res, self).__init__()
        self.__name__ = "C2D_3x3_128_RES"
        self.channels = channels
        self.filters_count = filters_count
        
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C2D_Res(self.filters_count[0], 3),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C2D_Res(self.filters_count[1], 3),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C2D_Res(self.filters_count[2], 3),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
            C2D_Res(self.filters_count[3], 3),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 4, 1, conv_type = conv_type, activation_type = "tanh"),
#             C2D_BN_A(self.filters_count[4], self.filters_count[5], 2, 1, conv_type = conv_type, activation_type = "tanh"),
        )
        
        self.decoder = nn.Sequential(
#             CT2D_BN_A(self.filters_count[5], self.filters_count[4], 2, 1),
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 4, 1),
            CT2D_Res(self.filters_count[3], 3),
            CT2D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2),
            CT2D_Res(self.filters_count[2], 3),
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            CT2D_Res(self.filters_count[1], 3),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 4, 2),
            CT2D_Res(self.filters_count[0], 3),
            CT2D_BN_A(self.filters_count[0], self.filters_count[0], 4, 2),
            C2D_BN_A(self.filters_count[0], self.channels, 3, 1, activation_type="sigmoid")
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class C2D_AE_ACB_128_3x3(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,64,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_ACB_128_3x3, self).__init__()
        self.__name__ = "C2D_3x3_128_ACB"
        self.channels = channels
        self.filters_count = filters_count
        
        self.encoder = nn.Sequential(
            C2D_ACB(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C2D_ACB(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C2D_ACB(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C2D_ACB(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
            C2D_ACB(self.filters_count[3], self.filters_count[4], 3, 2, conv_type = conv_type, activation_type = "tanh"),
#             C2D_BN_A(self.filters_count[4], self.filters_count[5], 3, 1, conv_type = conv_type, activation_type = "tanh"),
        )
        
        self.decoder = nn.Sequential(
#             CT2D_BN_A(self.filters_count[5], self.filters_count[4], 2, 1),
            CT2D_ADB(self.filters_count[4], self.filters_count[3], 4, 1),
            CT2D_ADB(self.filters_count[3], self.filters_count[2], 3, 2),
            CT2D_ADB(self.filters_count[2], self.filters_count[1], 3, 2),
            CT2D_ADB(self.filters_count[1], self.filters_count[0], 3, 2),
            CT2D_ADB(self.filters_count[0], self.filters_count[0], 6, 2),
            C2D_BN_A(self.filters_count[0], self.channels, 3, 1, activation_type = "sigmoid")
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class C2D_AE_ACB_128_5x5(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,64,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_ACB_128_5x5, self).__init__()
        self.__name__ = "C2D_5x5_128_ACB"
        self.channels = channels
        self.filters_count = filters_count[1:]
        
        self.encoder = nn.Sequential(
            C2D_ACB(self.channels, self.filters_count[0], 5, 2, conv_type = conv_type),
            C2D_ACB(self.filters_count[0], self.filters_count[1], 5, 2, conv_type = conv_type),
            C2D_ACB(self.filters_count[1], self.filters_count[2], 5, 2, conv_type = conv_type),
            C2D_ACB(self.filters_count[2], self.filters_count[3], 5, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 5, 1, conv_type = conv_type, activation_type = "tanh"),
        )
        
        self.decoder = nn.Sequential(
            CT2D_ADB(self.filters_count[4], self.filters_count[3], 3, 1),
            CT2D_ADB(self.filters_count[3], self.filters_count[2], 4, 2),
            CT2D_ADB(self.filters_count[2], self.filters_count[1], 5, 2),
            CT2D_ADB(self.filters_count[1], self.filters_count[0], 5, 2),
            CT2D_ADB(self.filters_count[0], self.filters_count[0], 5, 2),
            C2D_BN_A(self.filters_count[0], self.channels, 6, 1, activation_type = "sigmoid")
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class C2D_AE_128_PC(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,64,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_PC, self).__init__()
        self.__name__ = "C2D_3x3_128_PC"
        self.channels = channels
        self.filters_count = filters_count[2:]
        
        self.el1 = C2D_BN_A(self.channels, self.filters_count[0]//2, 3, 2, conv_type = conv_type)
        self.er1 = C2D_BN_A(self.channels, self.filters_count[0]//2, 4, 2, conv_type = conv_type)
        
        self.el2 = C2D_BN_A(self.filters_count[0], self.filters_count[1]//2, 3, 2, conv_type = conv_type)
        self.er2 = C2D_BN_A(self.filters_count[0], self.filters_count[1]//2, 4, 2, padding = 1, conv_type = conv_type)
        
        self.el3 = C2D_BN_A(self.filters_count[1], self.filters_count[2]//2, 5, 3, conv_type = conv_type)
        self.er3 = C2D_BN_A(self.filters_count[1], self.filters_count[2]//2, 7, 3, conv_type = conv_type)
        
        self.el4 = C2D_BN_A(self.filters_count[2], self.filters_count[3]//2, 3, 2, conv_type = conv_type, activation_type=False)
        self.er4 = C2D_BN_A(self.filters_count[2], self.filters_count[3]//2, 4, 2, padding = 1, conv_type = conv_type, activation_type=False)
        
        self.e_act = nn.Tanh()
        
        self.dl1 = CT2D_BN_A(self.filters_count[3], self.filters_count[2]//2, 4, 2)
        self.dr1 = CT2D_BN_A(self.filters_count[3], self.filters_count[2]//2, 6, 2, padding = 1)
        
        self.dl2 = CT2D_BN_A(self.filters_count[2], self.filters_count[1]//2, 5, 3)
        self.dr2 = CT2D_BN_A(self.filters_count[2], self.filters_count[1]//2, 7, 3, padding = 1)
        
        self.dl3 = CT2D_BN_A(self.filters_count[1], self.filters_count[0]//2, 3, 2)
        self.dr3 = CT2D_BN_A(self.filters_count[1], self.filters_count[0]//2, 5, 2, padding = 1)
        
        self.dl4 = CT2D_BN_A(self.filters_count[0], self.filters_count[0]//2, 3, 2)
        self.dr4 = CT2D_BN_A(self.filters_count[0], self.filters_count[0]//2, 5, 2, padding = 1)
        
        self.d5 = CT2D_BN_A(self.filters_count[0], self.filters_count[0], 3, 1)
        self.c6 = C2D_BN_A(self.filters_count[0], self.channels, 6, 1, activation_type = "sigmoid")
        
    def forward(self, x):
        eo1 = torch.cat((self.el1(x), self.er1(x)), dim = 1)
        eo2 = torch.cat((self.el2(eo1), self.er2(eo1)), dim = 1)
        eo3 = torch.cat((self.el3(eo2), self.er3(eo2)), dim = 1)
        encodings = self.e_act(torch.cat((self.el4(eo3), self.er4(eo3)), dim = 1))
        
        do1 = torch.cat((self.dl1(encodings), self.dr1(encodings)), dim = 1)
        do2 = torch.cat((self.dl2(do1), self.dr2(do1)), dim = 1)
        do3 = torch.cat((self.dl3(do2), self.dr3(do2)), dim = 1)
        do4 = torch.cat((self.dl4(do3), self.dr4(do3)), dim = 1)
        do5 = self.d5(do4)
        reconstructions = self.c6(do5)
        return reconstructions, encodings

class C2D_Multi_AE(nn.Module):
    def __init__(self,
                 channels = 3,
                 filters_count = [64,64,96,96,128], 
                 conv_type = "conv2d"
                ):
        super(C2D_Multi_AE, self).__init__()
        self.channels = channels
        self.filters_count = filters_count
        self.__name__ = "C2D_Multi"
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 3, 1, activation_type="tanh"),
        )
        
        self.decoder = nn.Sequential(
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 3, 1),
            CT2D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2),
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 3, 2),
            CT2D_BN_A(self.filters_count[0], self.channels, 4, 2, activation_type="sigmoid"),
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings

class C2D_Multi_VAE(C2D_Multi_AE):
    def __init__(
        self,
        image_size,
        isTrain = True,
        channels = 3,
        filters_count = [64,64,96,96,128],
        conv_type = "conv2d"
    ):
        C2D_Multi_AE.__init__(self, channels = channels, filters_count = filters_count, conv_type = conv_type)
        self.__name__ = "C2D_Multi_VAE"
        self.image_size = image_size
        self.embedding_dim = list(self.encoder(torch.rand(1, self.channels, self.image_size, self.image_size)).shape)
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
    
    def vae_loss(self, original, reconstruction, mu, logvar, variational_beta = 0.9):
        recon_loss = F.binary_cross_entropy(reconstruction.flatten(), original.flatten(), reduction='sum')
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

class C2D_DP_AE_128_3x3(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,64,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_DP_AE_128_3x3, self).__init__()
        self.__name__ = "C2D_DP_3x3_128"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,128,4,4]
        
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            nn.Dropout2d(0.2),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            nn.Dropout2d(0.2),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
            nn.Dropout2d(0.2),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 4, 1, conv_type = conv_type, activation_type = "tanh"),
#             C2D_BN_A(self.filters_count[4], self.filters_count[5], 2, 1, conv_type = conv_type, activation_type = "tanh"),
        )
        
        self.decoder = nn.Sequential(
#             CT2D_BN_A(self.filters_count[5], self.filters_count[4], 2, 1),
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 4, 1),
            CT2D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2),
            nn.Dropout2d(0.2),
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            nn.Dropout2d(0.2),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 4, 2),
            nn.Dropout2d(0.2),
            CT2D_BN_A(self.filters_count[0], self.filters_count[0], 4, 2),
            C2D_BN_A(self.filters_count[0], self.channels, 3, 1, activation_type = "sigmoid")
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings

C2D_MODELS_DICT = {
    128: {
        "vanilla": {
            "3x3": C2D_AE_128_3x3,
            "5x5": C2D_AE_128_5x5
        },
        "acb": {
            "3x3": C2D_AE_ACB_128_3x3,
            "5x5": C2D_AE_ACB_128_5x5
        },
        "vae": {
            "3x3": C2D_AE_128_3x3_VAE,
            "5x5": C2D_AE_128_5x5_VAE,
            "multi_resolution": C2D_Multi_VAE
        },
        "res": {
            "3x3": C2D_AE_3x3_Res
        },
        "parallel": {
            "3x3": C2D_AE_128_PC
        },
        "dropout": {
            "3x3": C2D_DP_AE_128_3x3
        }
    },
    
    224: {
        "generic": Generic_C2D_AE
    },
    
    "multi_resolution": C2D_Multi_AE
}