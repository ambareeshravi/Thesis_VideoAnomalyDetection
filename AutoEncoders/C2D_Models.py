from attention_conv import *

# Models
class Generic_C2D_AE(nn.Module):
    def __init__(self,
                 # out_channels, kernel_size, stride, padding, dilation
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
        self.__name__ = "C2D_AE_Generic_Generic-"
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
                decoder_layers.append(nn.ConvTranspose2d(in_channels, self.channels, kernel_size if kernel_size%2==0 else kernel_size + 1, stride, padding = padding, output_padding = output_padding))
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
        encoder_activation = "tanh",
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_3x3, self).__init__()
        self.__name__ = "C2D_AE_128_3x3-"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[4],4,4]
        
        self.encoder = nn.Sequential(
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 4, 1, conv_type = conv_type, activation_type = encoder_activation),
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
        self.__name__ = "C2D_AE_128_5x5-"
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
        self.__name__ = "C2D_AE_128_3x3_VAE-"
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
        self.__name__ = "C2D_AE_128_5x5_VAE-"
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

class C2D_AE_128_3x3_Res(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_3x3_Res, self).__init__()
        self.__name__ = "C2D_AE_128_3x3_RES-"
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
    
class C2D_AE_128_3x3_ACB(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_3x3_ACB, self).__init__()
        self.__name__ = "C2D_AE_128_3x3_ACB-"
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
    
class C2D_AE_128_5x5_ACB(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_5x5_ACB, self).__init__()
        self.__name__ = "C2D_AE_128_5x5_ACB-"
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
    
class C2D_AE_128_Multi_PC(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_Multi_PC, self).__init__()
        self.__name__ = "C2D_AE_128_MULTI_PC-"
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

class C2D_AE_Multi_3x3(nn.Module):
    def __init__(self,
                 image_size = 128,
                 channels = 3,
                 filters_count = [64,64,96,96,128,128,256,256], 
                 conv_type = "conv2d"
                ):
        super(C2D_AE_Multi_3x3, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        self.__name__ = "C2D_AE_MULTI_3x3-"
        
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
    
class C2D_AE_Multi_3x3_VAE(C2D_AE_Multi_3x3):
    def __init__(
        self,
        image_size = 128,
        isTrain = True,
        channels = 3,
        conv_type = "conv2d"
    ):
        C2D_AE_Multi_3x3.__init__(self, image_size = image_size, channels = channels, conv_type = conv_type)
        self.__name__ = "C2D_AE_MULTI_3x3_VAE-"
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

class C2D_AE_128_3x3_DP(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_3x3_DP, self).__init__()
        self.__name__ = "C2D_AE_128_3x3_DP-"
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

class C2D_AE_128_3x3_AAC(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,64,128],
    ):
        super(C2D_AE_128_3x3_AAC, self).__init__()
        self.__name__ = "C2D_AE_128_3x3_AAC-"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[4],4,4]
        
        self.encoder = nn.Sequential(
            AugmentedConv(in_channels=self.channels, out_channels=self.filters_count[0], kernel_size=3, dk = self.filters_count[0]//8, dv = 1, Nh = 1,  stride = 2),
            nn.BatchNorm2d(self.filters_count[0]),
            nn.ReLU(),
            AugmentedConv(in_channels=self.filters_count[0], out_channels=self.filters_count[1], kernel_size=3, dk = self.filters_count[1]//8, dv = 1, Nh = 1,  stride = 2),
            nn.BatchNorm2d(self.filters_count[1]),
            nn.ReLU(),
            AugmentedConv(in_channels=self.filters_count[1], out_channels=self.filters_count[2], kernel_size=3, dk = self.filters_count[2]//8, dv = 1, Nh = 1,  stride = 2),
            nn.BatchNorm2d(self.filters_count[2]),
            nn.ReLU(),
            AugmentedConv(in_channels=self.filters_count[2], out_channels=self.filters_count[3], kernel_size=3, dk = self.filters_count[4]//8, dv = 1, Nh = 1,  stride = 2),
            nn.BatchNorm2d(self.filters_count[3]),
            nn.ReLU(),
            nn.Conv2d(self.filters_count[3], self.filters_count[4], 5, 1),
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
        self.__name__ = "C2D_AE_128_WIDE-"
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

class C2D_AE_128_3x3_SQZEX(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_3x3_SQZEX, self).__init__()
        self.__name__ = "C2D_AE_128_3x3_SQZEX-"
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

class C2D_AE_128_3x3_DoubleHead(nn.Module):
    def __init__(
        self,
        image_channels = 3,
        flow_channels = 3, 
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_3x3_DoubleHead, self).__init__()
        self.__name__ = "C2D_AE_128_3x3_DOUBLEHEAD-"
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
    
class C2D_AE_128_3x3_ALW(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        encoder_activation = "tanh",
        conv_type = "conv2d",
        lambda_ = 1e-3
    ):
        super(C2D_AE_128_3x3_ALW, self).__init__()
        self.__name__ = "C2D_AE_128_3x3_ALW-"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[4],4,4]
        self.lambda_ = lambda_
        
        self.encoder = nn.Sequential(
            ConvAttentionLayerWrapper(
                C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
                in_channels=self.channels,
                out_channels=self.filters_count[0],
                stride = 2
            ),
            ConvAttentionLayerWrapper(
                C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
                in_channels=self.filters_count[0],
                out_channels=self.filters_count[1],
                stride = 2
            ),
            ConvAttentionLayerWrapper(
                C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
                in_channels = self.filters_count[1],
                out_channels = self.filters_count[2],
                stride = 2
            ),
            ConvAttentionLayerWrapper(
                C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
                in_channels = self.filters_count[2],
                out_channels = self.filters_count[3],
                stride = 2
            ),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 4, 1, conv_type = conv_type, activation_type = encoder_activation),
        )
        
        self.decoder = nn.Sequential(
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 4, 1),
            ConvTransposeAttentionLayerWrapper(
                CT2D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2),
                in_channels=self.filters_count[3],
                out_channels=self.filters_count[2],
                stride = 2
            ),
            ConvTransposeAttentionLayerWrapper(
                CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
                in_channels = self.filters_count[2],
                out_channels = self.filters_count[1],
                stride = 2
            ),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 4, 2),
            CT2D_BN_A(self.filters_count[0], self.filters_count[0], 4, 2),
            C2D_BN_A(self.filters_count[0], self.channels, 3, 1, activation_type = "sigmoid")
        )
    
    def get_attention_loss(self):
        attention_losses = list()
        for layer in self.encoder:
            try: attention_losses.append(layer.attention_loss())
            except: continue
        for layer in self.decoder:
            try: attention_losses.append(layer.attention_loss())
            except: continue
        return self.lambda_ * torch.stack(attention_losses).sum()
        
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class C2D_AE_128_3x3_OriginPush(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_3x3_OriginPush, self).__init__()
        self.__name__ = "C2D_AE_128_3x3_ORIGINPUSH-"
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
    
class C2D_AE_224_5x5(nn.Module):
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
        super(C2D_AE_224_5x5, self).__init__()
        self.__name__ = "C2D_AE_224_5x5"
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
            self.__name__ += "_RES"
        if add_sqzex:
            self.__name__ += "_SQZEX"
        self.__name__ += "-"
        
        assert (add_res and add_sqzex) != True, "Either Squeeze Excitation Block or Residual Block. Not both"
        
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
        
    @staticmethod
    def get_AAC(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 0, conv_type = None, activation_type="leaky_relu"):
        return nn.Sequential(
            AugmentedConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            BN_A(out_channels, activation_type = activation_type, is3d=False)
        )
    
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
class C2D_AE_224_5x5_VAE(C2D_AE_224_5x5):
    def __init__(
        self,
        isTrain = True,
        channels = 3,
        filters_count = [64,64,96,96,128],
        conv_type = "conv2d"
    ):
        C2D_AE_224_5x5.__init__(self, channels = channels, filters_count = filters_count, conv_type = conv_type)
        self.__name__ = "C2D_AE_224_5x5_VAE-"
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
    
class C2D_AE_224_5x5_ACB(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,96,96,128],
        conv_type = "conv2d",
        encoder_activation = "tanh"
    ):
        super(C2D_AE_224_5x5_ACB, self).__init__()
        self.__name__ = "C2D_AE_224_5x5_ACB-"
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
        
class C2D_AE_COMBO(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        
        filter_sizes = [3] * 5,
        filter_strides = [2] * 5,
        filter_count = [64,64,64,96,96],
        conv_layers = ["conv2d"] * 5,
        convt_layers = ["conv2d_transpose"] * 5,
        acb_layers = [False] * 5,
        res_layers = [False] * 5,
        sqzex_layers = [False] * 5,
        dropouts = [0.0] * 5,
        
        encoder_activation = "relu",
        decoder_activation = "relu",
        encoding_activation = "tanh",
        reconstruction_activation = "sigmoid",
        stop_size = None,
        extra_id = "",
        useInputAttention = False,
        useAAC_first = False,
        isDecoderReplica = True,
        addEndConv = True
    ):
        super(C2D_AE_COMBO, self).__init__()
        self.__name__ = "C2D_AE_%d_%dx%d_COMBO_%s"%(image_size, filter_sizes[0], filter_sizes[0], extra_id)
        self.image_size = image_size
        self.channels = channels
        
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.filter_count = filter_count
        self.conv_layers = conv_layers
        self.convt_layers = convt_layers
        self.acb_layers = acb_layers
        self.res_layers = res_layers
        self.sqzex_layers = sqzex_layers
        self.dropouts = dropouts
        
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.encoding_activation = encoding_activation
        self.stop_size = stop_size
        self.useInputAttention = useInputAttention
        self.useAAC_first = useAAC_first
        self.isDecoderReplica = isDecoderReplica
        self.addEndConv = addEndConv
        
        self.image_sizes = [self.image_size]
        
        if self.stop_size == None: self.stop_size = (self.filter_sizes[0] * 3) - (self.filter_sizes[0] // 2)
        
        encoder_layers, decoder_layers = list(), list()
        
        # Build Encoder
        i_c = channels
        i_size = self.image_size
        endLoop = False
        active_layer = 0
        for idx, (k, s, o_c, c_layer, isACB, isRes, isSqzex, dp, ) in enumerate(zip(
            self.filter_sizes,
            self.filter_strides,
            self.filter_count,
            self.conv_layers,
            self.acb_layers,
            self.res_layers,
            self.sqzex_layers,
            self.dropouts,
        )):
            if endLoop: break
            active_layer = idx
            endLoop = (i_size <= self.stop_size) or (idx + 1) >= len(filter_count)
            if endLoop and ~(k%2):
                k += 1
                s = 1
            i_size = getConvOutputShape(i_size, k, s)
            self.image_sizes.append([i_size,k,s])
            
            layer = None
            if ~idx:
                if self.useAAC_first and not self.useInputAttention:
                    layer = get_AAC_BNA(in_channels = i_c, out_channels = o_c, kernel_size = k, stride = s, padding = 0, conv_type = c_layer, activation_type=self.encoder_activation)
                    
            if layer == None:
                layer_variant = C2D_ACB if isACB else C2D_BN_A
                layer = layer_variant(in_channels = i_c, out_channels = o_c, kernel_size = k, stride = s, padding = 0, activation_type = self.encoder_activation, conv_type = c_layer)
                if (idx<1) and not self.useAAC_first and self.useInputAttention:
                    layer.channels = i_c
                    layer.__name__ = "ip_cov"
                    layer = SoftMaxConvAttentionWrapper5(layer, layerWrapper=True)
            encoder_layers.append(layer)
            if isRes: encoder_layers.append(C2D_Res(channels = o_c, kernel_size = k, activation_type = self.encoder_activation, conv_type = c_layer))
            if isSqzex: encoder_layers.append(SE_Block(o_c))
            if dp: encoder_layers.append(nn.Dropout2d(dp))

            i_c = o_c
                
        # Build decoder
        for idx, (k, s, o_c, ct_layer, isACB, isRes, isSqzex, dp) in enumerate(zip(
            self.filter_sizes[:len(encoder_layers)][::-1],
            self.filter_strides[:len(encoder_layers)][::-1],
            self.filter_count[:len(encoder_layers)][::-1],
            self.convt_layers[:len(encoder_layers)][::-1],
            self.acb_layers[:len(encoder_layers)][::-1],
            self.res_layers[:len(encoder_layers)][::-1],
            self.sqzex_layers[:len(encoder_layers)][::-1],
            self.dropouts[:len(encoder_layers)][::-1],
        )):
            if (idx == 0) and ~(k%2): k += 1; s = 1
            if (idx >= active_layer) and (k%2 != 0):
                k += 1
                if self.addEndConv:
                    k += 2
            i_size = getConvTransposeOutputShape(i_size, k, s)
            self.image_sizes.append([i_size,k,s])
            layer_variant = CT2D_ADB if isACB else CT2D_BN_A
            layer = layer_variant(in_channels = i_c, out_channels = o_c, kernel_size = k, stride = s, padding = 0, activation_type = self.decoder_activation, conv_type = ct_layer)
                
            decoder_layers.append(layer)
            try:
                if isRes and self.isDecoderReplica: decoder_layers.append(CT2D_Res(channels = o_c, kernel_size = k, activation_type = self.decoder_activation, conv_type = ct_layer))
            except:
                pass
            if isSqzex and self.isDecoderReplica: decoder_layers.append(SE_Block(o_c))
            if dp: decoder_layers.append(nn.Dropout2d(dp))

            i_c = o_c
            
        if self.addEndConv:
            k = self.filter_sizes[0]
            decoder_layers.append(
                C2D_BN_A(in_channels = o_c, out_channels = self.channels, kernel_size = k, stride = 1, activation_type=reconstruction_activation)
            )
            i_size = getConvOutputShape(i_size, k, 1)
            self.image_sizes.append([i_size, k, 1])
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings

# Models Dic
C2D_MODELS_DICT = {
    128: {
        "vanilla": {
            "3x3": C2D_AE_128_3x3,
            "5x5": C2D_AE_128_5x5
        },
        "acb": {
            "3x3": C2D_AE_128_3x3_ACB,
            "5x5": C2D_AE_128_5x5_ACB
        },
        "vae": {
            "3x3": C2D_AE_128_3x3_VAE,
            "5x5": C2D_AE_128_5x5_VAE,
            "multi_resolution": C2D_AE_Multi_3x3_VAE
        },
        "res": {
            "3x3": C2D_AE_128_3x3_Res
        },
        "parallel": {
            "3x3": C2D_AE_128_Multi_PC
        },
        "dropout": {
            "3x3": C2D_AE_128_3x3_DP
        },
        "wide": {
            "3x3": C2D_AE_128_WIDE
        },
        "squeeze_excitation": {
            "3x3": C2D_AE_128_3x3_SQZEX
        },
        "double_head": {
            "3x3": C2D_AE_128_3x3_DoubleHead
        }
    },
    
    224: {
        "generic": Generic_C2D_AE,
        "vanilla": C2D_AE_224_5x5,
        "acb": C2D_AE_224_5x5_ACB,
        "vae": C2D_AE_224_5x5_VAE,
    },
    
    "multi_resolution": C2D_AE_Multi_3x3,
    "best": C2D_AE_COMBO
}


class C2D_AE_128_LC(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,96,96,128],
        encoder_activation = "tanh",
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_LC, self).__init__()
        self.__name__ = "C2D_AE_128_LayerConnected-"
        self.channels = channels
        self.filters_count = filters_count 
        self.embedding_dim = [1,self.filters_count[4],4,4]
        
        self.encoder_layers = [
            C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 4, 1, conv_type = conv_type, activation_type = encoder_activation),
        ]
        
        self.decoder_layers = [
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 4, 1),
            CT2D_BN_A(self.filters_count[3] * 2, self.filters_count[2], 3, 2),
            CT2D_BN_A(self.filters_count[2] * 2, self.filters_count[1], 3, 2),
#             CT2D_BN_A(self.filters_count[1] * 2, self.filters_count[0], 4, 2),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 4, 2),
            CT2D_BN_A(self.filters_count[0], self.filters_count[0], 4, 2),
            C2D_BN_A(self.filters_count[0], self.channels, 3, 1, activation_type = "sigmoid")
        ]
        
        self.layers = nn.Sequential(*(self.encoder_layers + self.decoder_layers))
        
    def forward(self, x):
        e1 = self.encoder_layers[0](x) # 64x63x63
        e2 = self.encoder_layers[1](e1) # 64x31x31
        e3 = self.encoder_layers[2](e2) # 96x15x15
        e4 = self.encoder_layers[3](e3) # 96x7x7
        e5 = self.encoder_layers[4](e4) # 128x4x4
        
        d1 = self.decoder_layers[0](e5) # 96x7x7
        d2 = self.decoder_layers[1](torch.cat((d1, e4), dim = 1)) # 96x15x15
        d3 = self.decoder_layers[2](torch.cat((d2, e3), dim = 1)) # 64x31x31
#         d4 = self.decoder_layers[3](torch.cat((d3, e2), dim = 1)) # 64x63x63
        d4 = self.decoder_layers[3](d3) # 64x63x63
        d5 = self.decoder_layers[4](d4) # 64x130x130
        d6 = self.decoder_layers[5](d5) # cx128x128
        
        encodings = e5
        reconstructions = d6
        return reconstructions, encodings
    
class UCSD1_AE(nn.Module):
    def __init__(self, filters_count = [512,256,128,128,64]):
        super(UCSD1_AE, self).__init__()
        self.__name__ = "UCSD1_AE"
        self.channels = 1
        self.filters_count = filters_count
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.filters_count[0], kernel_size=(3,4), stride=1),
            nn.BatchNorm2d(self.filters_count[0]),
            nn.ReLU(),
            nn.Conv2d(self.filters_count[0], self.filters_count[1], (3,4), stride=1),
            nn.BatchNorm2d(self.filters_count[1]),
            nn.ReLU(),
            nn.AvgPool2d((2,2)),
            nn.Conv2d(self.filters_count[1], self.filters_count[2], (3,4), stride=2),
            nn.BatchNorm2d(self.filters_count[2]),
            nn.ReLU(),
            nn.AvgPool2d((2,2)),
            nn.Conv2d(self.filters_count[2], self.filters_count[3], (3,4), stride=2),
            nn.BatchNorm2d(self.filters_count[3]),
            nn.ReLU(),
            nn.Conv2d(self.filters_count[3], self.filters_count[4], (3,4), stride=2),
            nn.BatchNorm2d(self.filters_count[4]),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.filters_count[4], self.filters_count[3], (3,5), stride=2),
            nn.BatchNorm2d(self.filters_count[3]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.filters_count[3], self.filters_count[2], (3,4), stride=2),
            nn.BatchNorm2d(self.filters_count[2]),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=(2,2)),
            nn.ConvTranspose2d(self.filters_count[2], self.filters_count[1], (3,4), stride=2),
            nn.BatchNorm2d(self.filters_count[1]),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=(2,2)),
            nn.ConvTranspose2d(self.filters_count[1], self.filters_count[0], (3,5), stride=1),
            nn.BatchNorm2d(self.filters_count[0]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.filters_count[0], self.channels, (3,7), stride=1),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings
    
    
# Experimental

def axis_normalize(x, axis):
    x -= x.min(axis, keepdim = True)[0]
    x /= x.max(axis, keepdim = True)[0]
    return x

class SMCAW5_layer(nn.Module):
    def __init__(self, model, kernel_sizes = (3,5), projection = 64, channels = 64):
        super(SMCAW5_layer, self).__init__()
        self.model = model
        self.projection = projection
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.channels, self.projection, self.kernel_sizes[0], 1, padding = self.kernel_sizes[0]//2),
            nn.Conv2d(self.projection, self.channels, self.kernel_sizes[1], 1, padding = self.kernel_sizes[1]//2),
            nn.BatchNorm2d(self.channels),
        )
        self.max_pool = nn.MaxPool2d(9, stride = 1, padding = 4)
        
    def attention_forward(self, x):
        attention_activations = self.attention_conv(x)
        v = torch.sum(attention_activations, keepdim = True, axis = -2)
        h = torch.sum(attention_activations, keepdim = True, axis = -1)
        vs = axis_normalize(v, axis = -1)
        hs = axis_normalize(h, axis = -2)

        # Normalization to improve visualization
        vs = scale(vs, t_min = 3e-1)
        hs = scale(hs, t_min = 3e-1)
        x_a = torch.matmul(hs, vs)
        x_a = self.max_pool(x_a)
        self.xam = x_a
        return torch.multiply(x, x_a)
        
    def forward(self, x, returns = 1):
        x_a = self.attention_forward(x)
        model_returns = self.model(x_a)
        if returns == 1: return model_returns
        if returns == 2: return tuple(list(model_returns) + [x_a])
        if returns == 3: return tuple(list(model_returns) + [x_a, self.xam])        
        
class C2D_AE_128_3x3_AATW(nn.Module):
    def __init__(
        self,
        channels = 3,
        filters_count = [64,64,64,96,96,128],
        encoder_activation = "tanh",
        conv_type = "conv2d"
    ):
        super(C2D_AE_128_3x3_AATW, self).__init__()
        self.__name__ = "C2D_AE_128_3x3_AATW-"
        self.channels = channels
        self.filters_count = filters_count
        self.embedding_dim = [1,self.filters_count[4],4,4]
        
        self.encoder = nn.Sequential(
            SMCAW5_layer(
                C2D_BN_A(self.channels, self.filters_count[0], 3, 2, conv_type = conv_type),
                projection = min(64, self.filters_count[0]),
                channels = self.channels,
            ),
            SMCAW5_layer(
                C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2, conv_type = conv_type),
                projection = min(64, self.filters_count[1]),
                channels = self.filters_count[0],
            ),
            SMCAW5_layer(
                C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2, conv_type = conv_type),
                projection = min(64, self.filters_count[2]),
                channels = self.filters_count[1],
            ),
            SMCAW5_layer(
                C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2, conv_type = conv_type),
                projection = min(64, self.filters_count[3]),
                channels = self.filters_count[2],
            ),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 4, 1, conv_type = conv_type, activation_type = encoder_activation),
        )
        
        self.decoder = nn.Sequential(
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


# C2D_AE_BEST(
#     image_size = 128,
#     channels = 3,
#     filters_count = [64,64,96,96,128],
#     encoder_activation = "tanh",
#     conv_type = "conv2d",
    
#     kernel_size = 3,
#     stride = 2,
#     stop_size = None,
#     final_kernel_size = None,
    
#     use_aug_conv = False,
#     use_input_attention = False,
#     add_sqzex = False,
#     add_dropouts = False,
#     add_res = False,
    
#     dropout_rate = 0.2
# )        