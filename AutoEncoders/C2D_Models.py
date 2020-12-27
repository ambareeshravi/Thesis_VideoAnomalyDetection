import sys
sys.path.append("..")

from torch.nn import functional as F

from general import *
from general.model_utils import *
from general.losses import max_norm

from attention_conv import AugmentedConv

# [64,64,96,96,128,128]

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
        filters_count = [64,64,64,96,96,128],
        encoding_activation = "tanh",
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_3x3, self).__init__()
        self.__name__ = "C2D_3x3_128"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[4],4,4]
        
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 4, 1, conv_type = conv_type, activation_type = encoding_activation),
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
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_5x5, self).__init__()
        self.__name__ = "C2D_5x5_128"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[4],4,4]
        
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
        filters_count = [64,64,64,96,96,128],
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
        filters_count = [64,64,64,96,96,128],
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
        filters_count = [64,64,64,96,96,128],
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
        filters_count = [64,64,64,96,96,128],
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
        filters_count = [64,64,64,96,96,128],
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
        filters_count = [64,64,64,96,96,128],
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
    
    def encoder(self, x):
        eo1 = torch.cat((self.el1(x), self.er1(x)), dim = 1)
        eo2 = torch.cat((self.el2(eo1), self.er2(eo1)), dim = 1)
        eo3 = torch.cat((self.el3(eo2), self.er3(eo2)), dim = 1)
        encodings = self.e_act(torch.cat((self.el4(eo3), self.er4(eo3)), dim = 1))
        return encodings
    
    def decoder(self, encodings):
        do1 = torch.cat((self.dl1(encodings), self.dr1(encodings)), dim = 1)
        do2 = torch.cat((self.dl2(do1), self.dr2(do1)), dim = 1)
        do3 = torch.cat((self.dl3(do2), self.dr3(do2)), dim = 1)
        do4 = torch.cat((self.dl4(do3), self.dr4(do3)), dim = 1)
        do5 = self.d5(do4)
        reconstructions = self.c6(do5)
        return reconstructions
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings

class C2D_Multi_AE(nn.Module):
    def __init__(self,
                 image_size = 128,
                 channels = 3,
                 filters_count = [64,64,96,96,128,128,256,256], 
                 conv_type = "conv2d"
                ):
        super(C2D_Multi_AE, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        self.__name__ = "C2D_Multi"
        
        encoder_layers = list()
        new_image_size = self.image_size
        new_in_channels = self.channels
        idx = 0
        while new_image_size > 10:
            encoder_layers.append(
                C2D_BN_A(new_in_channels, self.filters_count[idx], 3, 2)
            )
            new_image_size = getConvOutputShape(new_image_size, 3, 2)
            new_in_channels = self.filters_count[idx]
            idx += 1
        encoder_layers.append(
            C2D_BN_A(new_in_channels, self.filters_count[idx], 3, 1, activation_type="tanh")
        )
        new_in_channels = self.filters_count[idx]
        new_image_size = getConvOutputShape(new_image_size, 3, 1)
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = list()
        decoder_layers.append(
            CT2D_BN_A(new_in_channels, self.filters_count[idx-1], 3, 1)
        )
        new_image_size = getConvTransposeOutputShape(new_image_size, 3, 2)
        new_in_channels = self.filters_count[idx-1]
        idx -= 1
        while idx > 0:
            decoder_layers.append(
                CT2D_BN_A(new_in_channels, self.filters_count[idx-1], 3, 2)
            )
            new_image_size = getConvTransposeOutputShape(new_image_size, 3, 2)
            new_in_channels = self.filters_count[idx-1]
            idx -= 1
        decoder_layers.append(
            CT2D_BN_A(new_in_channels, self.channels, 4, 2, activation_type="sigmoid")
        )
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class C2D_Multi_VAE(C2D_Multi_AE):
    def __init__(
        self,
        image_size = 128,
        isTrain = True,
        channels = 3,
        conv_type = "conv2d"
    ):
        C2D_Multi_AE.__init__(self, image_size = image_size, channels = channels, conv_type = conv_type)
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
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_DP_AE_128_3x3, self).__init__()
        self.__name__ = "C2D_DP_3x3_128"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[4],4,4]
        
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

class AttentionWrapper(nn.Module):
    def __init__(self, image_size, model, projection_ratio = 4):
        super(AttentionWrapper, self).__init__()
        self.image_size = image_size # w,h
        self.model = model
        self.projection_dim = (self.image_size[0] * projection_ratio, self.image_size[1] * projection_ratio)
        self.__name__ = self.model.__name__ + "_ATTENTION"
        
        self.W_v = nn.Parameter(torch.randn((self.image_size[1], self.projection_dim[1]), requires_grad=True))
        self.W_h = nn.Parameter(torch.randn((self.projection_dim[0], self.image_size[0]), requires_grad=True))
        self.attention_activation = nn.Sigmoid()
    
    def attention_forward(self, x):
        x_a = torch.matmul(torch.matmul(x, self.W_v), torch.matmul(self.W_h, x))
        return self.attention_activation(torch.multiply(x, x_a))
    
    def attention_loss(self, x):
        return torch.sum(x**2)
        
    def forward(self, x):
        x_a = self.attention_forward(x)
        encodings = self.model.encoder(x_a)
        reconstructions = self.model.decoder(encodings)
        return reconstructions, encodings, x_a

class ConvAttentionWapper(nn.Module):
    def __init__(self, model, kernel_sizes = (3,5), projection = 64, lambda_ = 1e-6, max_norm_clip = 1):
        super(ConvAttentionWapper, self).__init__()
        self.model = model
        self.projection = projection
        self.kernel_sizes = kernel_sizes
        self.lambda_ = lambda_
        self.max_norm_clip = max_norm_clip
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.model.channels, self.projection, self.kernel_sizes[0], 1, padding = self.kernel_sizes[0]//2),
            nn.Conv2d(self.projection, self.model.channels, self.kernel_sizes[1], 1, padding = self.kernel_sizes[1]//2),
        )
        self.__name__ = self.model.__name__ + "_CONV_ATTENTION"
        self.act_block = nn.Sequential(
            nn.BatchNorm2d(self.model.channels),
            nn.Sigmoid()
        )
    
    def attention_forward(self, x):
#         x_a = self.attention_conv(x)
#         return self.act_block(x_a)
        x_a = self.attention_conv(x)
        return self.act_block(torch.multiply(x, x_a))
    
    def attention_loss(self, w):
        return self.lambda_ * torch.sqrt(torch.sum(w**2))
#         return torch.sum(max_norm(self.attention_conv[0].weight.data, self.max_norm_clip)) + torch.sum(max_norm(self.attention_conv[1].weight.data, self.max_norm_clip))
        
    def forward(self, x):
        x_a = self.attention_forward(x)
        encodings = self.model.encoder(x_a)
        reconstructions = self.model.decoder(encodings)
        return reconstructions, encodings, x_a

ConvAttentionWrapper = ConvAttentionWapper

class AAC_AE(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,128,128],
    ):
        super(AAC_AE, self).__init__()
        self.__name__ = "AAC_AE"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[4],4,4]
        
        self.encoder = nn.Sequential(
            AugmentedConv(in_channels=self.channels, out_channels=self.filters_count[0]//4, kernel_size=3, dk = self.filters_count[0]//8, dv = 1, Nh = 1,  stride = 2),
            nn.BatchNorm2d(self.filters_count[0]//4),
            nn.ReLU(),
            AugmentedConv(in_channels=self.filters_count[0]//4, out_channels=self.filters_count[1]//4, kernel_size=3, dk = self.filters_count[1]//8, dv = 1, Nh = 1,  stride = 2),
            nn.BatchNorm2d(self.filters_count[1]//4),
            nn.ReLU(),
            AugmentedConv(in_channels=self.filters_count[1]//4, out_channels=self.filters_count[2]//4, kernel_size=3, dk = self.filters_count[2]//8, dv = 1, Nh = 1,  stride = 2),
            nn.BatchNorm2d(self.filters_count[2]//4),
            nn.ReLU(),
            AugmentedConv(in_channels=self.filters_count[2]//4, out_channels=self.filters_count[3]//4, kernel_size=3, dk = self.filters_count[4]//8, dv = 1, Nh = 1,  stride = 2),
            nn.BatchNorm2d(self.filters_count[3]//4),
            nn.ReLU(),
            nn.Conv2d(self.filters_count[3]//4, self.filters_count[4], 5, 1),
            nn.BatchNorm2d(self.filters_count[4]),
            nn.Tanh(),
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
    
class C2D_AE_128_WIDE(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [48,48,96,96,128,128]
    ):
        super(C2D_AE_128_WIDE, self).__init__()
        self.__name__ = "C2D_AE_128_WIDE"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[4],4,4]
        
        self.encoder = nn.ModuleList([
            self.get_parallel_conv_blocks(self.channels, self.filters_count[0], [3,5,7,9], 2),
            self.get_parallel_conv_blocks(self.filters_count[0], self.filters_count[1], [3,5,7,9], 2),
            self.get_parallel_conv_blocks(self.filters_count[1], self.filters_count[2], [3,5,7], 2),
            self.get_parallel_conv_blocks(self.filters_count[2], self.filters_count[3], [3,5,], 2),
            self.get_parallel_conv_blocks(self.filters_count[3], self.filters_count[4], [4], 2),
        ])
        
        self.encoder_act_blocks = nn.ModuleList([
            BN_A(self.filters_count[0], is3d=False),
            BN_A(self.filters_count[1], is3d=False),
            BN_A(self.filters_count[2], is3d=False),
            BN_A(self.filters_count[3], is3d=False),
            BN_A(self.filters_count[4], is3d=False),
        ])
        
        self.decoder = nn.ModuleList([
            self.get_parallel_deconv_blocks(self.filters_count[4], self.filters_count[3], [3], 2, padding_factor = 1),
            self.get_parallel_deconv_blocks(self.filters_count[3], self.filters_count[2], [3,5,], 2),
            self.get_parallel_deconv_blocks(self.filters_count[2], self.filters_count[1], [3,5,7], 2),
            self.get_parallel_deconv_blocks(self.filters_count[1], self.filters_count[0], [3,5,7,9], 2),
            self.get_parallel_deconv_blocks(self.filters_count[0], self.channels, [4], 2, padding_factor = 4),
        ])
        
        self.decoder_act_blocks = nn.ModuleList([
            BN_A(self.filters_count[3], is3d=False),
            BN_A(self.filters_count[2], is3d=False),
            BN_A(self.filters_count[1], is3d=False),
            BN_A(self.filters_count[0], is3d=False),
            BN_A(self.channels, is3d=False),
        ])
        
    def get_parallel_conv_blocks(self, in_channels, out_channels, kernel_sizes, stride, padding_factor = 1):
        return nn.ModuleList([nn.Conv2d(in_channels, out_channels // len(kernel_sizes), k, stride, padding = (k-padding_factor)//2) for k in kernel_sizes])
        
    def get_parallel_deconv_blocks(self, in_channels, out_channels, kernel_sizes, stride, padding_factor = 2):
        return nn.ModuleList([nn.ConvTranspose2d(in_channels, out_channels // len(kernel_sizes), k, stride, padding = (k-padding_factor)//2) for k in kernel_sizes])
        
    def forward(self, x):
        layer_input = x
        
        for layer, layer_act in zip(self.encoder, self.encoder_act_blocks):
            layer_output = torch.cat([l(layer_input) for l in layer], dim = 1)
            layer_input = layer_act(layer_output)
            
        encodings = layer_input
                
        for layer, layer_act in zip(self.decoder, self.decoder_act_blocks):
            layer_output = torch.cat([l(layer_input) for l in layer], dim = 1)
            layer_input = layer_act(layer_output)
            
        reconstructions = layer_input
        return reconstructions, encodings

class C2D_AE_128_3x3_SE(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_3x3_SE, self).__init__()
        self.__name__ = "C2D_3x3_128_SE"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[4],4,4]
        
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            SE_Block(self.filters_count[0]),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            SE_Block(self.filters_count[1]),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            SE_Block(self.filters_count[2]),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
            SE_Block(self.filters_count[3]),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 4, 1, conv_type = conv_type, activation_type = "tanh"),
#             C2D_BN_A(self.filters_count[4], self.filters_count[5], 2, 1, conv_type = conv_type, activation_type = "tanh"),
        )
        
        self.decoder = nn.Sequential(
#             CT2D_BN_A(self.filters_count[5], self.filters_count[4], 2, 1),
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 4, 1),
            SE_Block(self.filters_count[3]),
            CT2D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2),
            SE_Block(self.filters_count[2]),
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            SE_Block(self.filters_count[1]),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 4, 2),
            SE_Block(self.filters_count[0]),
            CT2D_BN_A(self.filters_count[0], self.filters_count[0], 4, 2),
            C2D_BN_A(self.filters_count[0], self.channels, 3, 1, activation_type = "sigmoid")
        )
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings    

class C2D_DoubleHead(nn.Module):
    def __init__(
        self,
        image_channels = 3,
        flow_channels = 3, 
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_DoubleHead, self).__init__()
        self.__name__ = "C2D_DoubleHead"
        self.image_channels = image_channels
        self.flow_channels = flow_channels
        self.filters_count = filters_count
        
        self.image_encoder = nn.Sequential(
            C2D_BN_A(self.image_channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 4, 1, conv_type = conv_type),
        )
        
        self.flow_encoder = nn.Sequential(
            C2D_BN_A(self.flow_channels, self.filters_count[0], 5, 3, conv_type = conv_type),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 5, 3, conv_type = conv_type),
            C2D_BN_A(self.filters_count[1], self.filters_count[4], 4, 3, conv_type = conv_type),
        )
        
        self.image_decoder = nn.Sequential(
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 4, 1),
            CT2D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2),
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 3, 2),
            CT2D_BN_A(self.filters_count[0], self.image_channels, 4, 2),
        )
        
        self.flow_decoder = nn.Sequential(
            CT2D_BN_A(self.filters_count[4], self.filters_count[1], 5, 3),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 5, 3),
            CT2D_BN_A(self.filters_count[0], self.filters_count[0], 5, 3),
            C2D_BN_A(self.filters_count[0], self.flow_channels, 7, 1, conv_type = conv_type)
        )
        
        self.bottleneck_squeeze = nn.AdaptiveAvgPool2d(4)
        self.bottleneck_expand = nn.UpsamplingBilinear2d((4,8))
        
    def forward(self, x):
        image_encodings = self.image_encoder(x[:,:self.image_channels,...])
        flow_encodings = self.flow_encoder(x[:,self.image_channels:,...])
        
        encodings = self.bottleneck_squeeze(torch.cat((image_encodings, flow_encodings), dim = -1))
        decodings = self.bottleneck_expand(encodings)
        image_decodings, flow_decodings = torch.split(decodings, 4, dim = -1)
        
        image_reconstructions = self.image_decoder(image_decodings)
        flow_reconstructions = self.flow_decoder(flow_decodings)
        
        reconstructions = torch.cat((image_reconstructions, flow_reconstructions), dim = 1)
        return reconstructions, encodings

class C2D_AE_128_OriginPush(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_OriginPush, self).__init__()
        self.__name__ = "C2D_AE_128_OriginPush"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[4],4,4]
        
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 4, 1, conv_type = conv_type, activation_type = "sigmoid"),
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
    
class C2D_AE_224(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,96,96,128],
        conv_type = "conv2d",
        encoder_activation = "tanh",
        use_aug_conv = False,
        add_sqzex = False,
        add_dropouts = False,
        add_res = False
    ):
        super(C2D_AE_224, self).__init__()
        self.__name__ = "C2D_AE_224"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1, self.filters_count[4], 4, 4] # check and change
        self.conv_layer = C2D_BN_A
        if use_aug_conv:
            self.__name__ += "_AAC"
            self.conv_layer = self.get_AAC
            add_res = False # change
        if add_dropouts:
            self.__name__ += "_DP"
        if add_res:
            self.__name__ += "_Res"
        if add_res:
            self.__name__ += "_SE"
        
        assert add_res != add_sqzex, "Either Squeeze Excitation Block or Residual Block. Not both"
        
        self.encoder_layers = [
            self.conv_layer(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            self.conv_layer(self.filters_count[0], self.filters_count[1], 5, 2, conv_type = conv_type),
            self.conv_layer(self.filters_count[1], self.filters_count[2], 5, 2, conv_type = conv_type),
            self.conv_layer(self.filters_count[2], self.filters_count[3], 5, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 7 if use_aug_conv else 5, 2, conv_type = conv_type, activation_type = encoder_activation),
        ]
        
        if add_sqzex:
            modified_layers = list()
            for (layer, output_filters) in zip(self.encoder_layers, self.filters_count):
                modified_layers.append(layer)
                modified_layers.append(SE_Block(output_filters))
            self.encoder_layers = modified_layers[:-1]
            
        if add_res:
            modified_layers = list()
            for (layer, output_filters) in zip(self.encoder_layers, self.filters_count):
                modified_layers.append(layer)
                modified_layers.append(C2D_Res(output_filters, 5))
            self.encoder_layers = modified_layers[:-1]
            
        if add_dropouts:
            modified_layers = list()
            for layer in self.encoder_layers:
                modified_layers.append(layer)
                modified_layers.append(nn.Dropout2d(0.2))
            self.encoder_layers = modified_layers[:-1]
        
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        self.decoder_layers = [
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 5, 2),
            CT2D_BN_A(self.filters_count[3], self.filters_count[2], 5, 2),
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 6, 2),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 5, 2),
            CT2D_BN_A(self.filters_count[0], self.channels, 4, 2, activation_type = "sigmoid"),
        ]
        
        if add_sqzex:
            modified_layers = list()
            for layer, output_filters in zip(self.decoder_layers, self.filters_count[::-1][1:] + [self.channels]):
                modified_layers.append(layer)
                if output_filters != self.channels:
                    modified_layers.append(SE_Block(output_filters))
            self.decoder_layers = modified_layers
            
        if add_res:
            modified_layers = list()
            for layer, output_filters in zip(self.decoder_layers, self.filters_count[::-1][1:] + [self.channels]):
                modified_layers.append(layer)
                modified_layers.append(CT2D_Res(output_filters, 5))
            self.decoder_layers = modified_layers[:-1]
            
        if add_dropouts:
            modified_layers = list()
            for layer in self.decoder_layers:
                modified_layers.append(layer)
                modified_layers.append(nn.Dropout2d(0.2))
            self.decoder_layers = modified_layers[:-1]
        
        self.decoder = nn.Sequential(*self.decoder_layers)
    
    def get_AAC(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 0, conv_type = None, activation_type="leaky_relu"):
        return nn.Sequential(
            AugmentedConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            BN_A(out_channels, activation_type = activation_type, is3d=False)
        )
    
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class C2D_AE_224_VAE(C2D_AE_224):
    def __init__(
        self,
        isTrain = True,
        channels = 3,
        filters_count = [64,64,96,96,128],
        conv_type = "conv2d"
    ):
        C2D_AE_224.__init__(self, channels = channels, filters_count = filters_count, conv_type = conv_type)
        self.__name__ = "C2D_AE_224_VAE"
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
    
class C2D_AE_ACB_224(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,96,96,128],
        conv_type = "conv2d",
        encoder_activation = "tanh"
    ):
        super(C2D_AE_ACB_224, self).__init__()
        self.__name__ = "C2D_AE_224_ACB"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1, self.filters_count[4], 4, 4] # check and change
        
        self.encoder = nn.Sequential(
            C2D_ACB(self.channels, self.filters_count[0], 3, 2),
            C2D_ACB(self.filters_count[0], self.filters_count[1], 5, 2),
            C2D_ACB(self.filters_count[1], self.filters_count[2], 5, 2),
            C2D_ACB(self.filters_count[2], self.filters_count[3], 5, 2),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 5, 3, activation_type = encoder_activation),
        )
        
        self.decoder = nn.Sequential(
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 5, 2),
            CT2D_ADB(self.filters_count[3], self.filters_count[2], 5, 2),
            CT2D_ADB(self.filters_count[2], self.filters_count[1], 6, 2),
            CT2D_ADB(self.filters_count[1], self.filters_count[0], 5, 2),
            CT2D_ADB(self.filters_count[0], self.channels, 4, 2, activation_type = "sigmoid"),
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
        },
        "wide": {
            "3x3": C2D_AE_128_WIDE
        },
        "squeeze_excitation": {
            "3x3": C2D_AE_128_3x3_SE
        },
        "double_head": {
            "3x3": C2D_DoubleHead
        }
    },
    
    224: {
        "generic": Generic_C2D_AE,
        "vanilla": C2D_AE_224,
        "acb": C2D_AE_ACB_224,
        "vae": C2D_AE_224_VAE,
    },
    
    "multi_resolution": C2D_Multi_AE
}