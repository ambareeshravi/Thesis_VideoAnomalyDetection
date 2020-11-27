import sys
sys.path.append("..")

from general import *
from general.model_utils import *

class Generator64(nn.Module):
    def __init__(self, channels = 3, embedding_size  = 100, ngf = 64, ngpu = 1):
        super(Generator64, self).__init__()
        self.embedding_size = embedding_size
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(embedding_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator64(nn.Module):
    def __init__(self, channels = 3, ndf = 64, ngpu = 1):
        super(Discriminator64, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    

class Generator(nn.Module):
    def __init__(self, embedding_size = 256, channels = 3, filter_count = np.array([8,4,4,2,1])*64):
        super(Generator, self).__init__()
        self.embedding_size = embedding_size
        self.filter_count = filter_count
        self.channels = channels
        
        self.generator = nn.Sequential(
            CT2D_BN_A(in_channels = self.embedding_size, out_channels = self.filter_count[-1], kernel_size = 5, stride = 2, activation_type = "relu"),
            CT2D_BN_A(in_channels = self.filter_count[-1], out_channels = self.filter_count[-2], kernel_size = 3, stride = 2, activation_type = "relu"),
            CT2D_BN_A(in_channels = self.filter_count[-2], out_channels = self.filter_count[-3], kernel_size = 3, stride = 2, activation_type = "relu"),
            CT2D_BN_A(in_channels = self.filter_count[-3], out_channels = self.filter_count[-4], kernel_size = 3, stride = 2, activation_type = "relu"),
            CT2D_BN_A(in_channels = self.filter_count[-4], out_channels = self.channels, kernel_size = 4, stride = 2, activation_type="tanh"),
        )

    def forward(self, embeddings):
        return self.generator(embeddings)
    
class Discriminator(nn.Module):
    def __init__(self, channels = 3, filter_count = np.array([8,4,4,2,1])*64, classes = 1):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.filter_count = filter_count
        self.classes = classes
        self.discriminator = nn.Sequential(
            C2D_BN_A(in_channels = self.channels, out_channels = self.filter_count[0], kernel_size = 5, stride = 3),
            C2D_BN_A(in_channels = self.filter_count[0], out_channels = self.filter_count[1], kernel_size = 5, stride = 2),
            C2D_BN_A(in_channels = self.filter_count[1], out_channels = self.filter_count[2], kernel_size = 5, stride = 2),
            C2D_BN_A(in_channels = self.filter_count[2], out_channels = self.filter_count[3], kernel_size = 3, stride = 2),
            C2D_BN_A(in_channels = self.filter_count[3], out_channels = self.classes, kernel_size = 3, stride = 2),
            nn.Flatten(),
            nn.Sigmoid()
            )

    def forward(self, images):
        return self.discriminator(images)