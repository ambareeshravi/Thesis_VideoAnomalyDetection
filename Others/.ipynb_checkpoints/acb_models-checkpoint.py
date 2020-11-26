import numpy as np
import torch
from torch import nn
from torchvision import models

np.random.seed(0)
torch.manual_seed(0)

class Config:
    ImageSize = 128
    EmbeddingSize = 300
    channels = 1

def get_Activation(activation_type = "relu", negative_slope = 0.2):
	if "sigmoid" in activation_type.lower(): return nn.Sigmoid()
	elif "leaky" in activation_type.lower(): return nn.LeakyReLU(negative_slope = negative_slope, inplace = True)
	elif "relu" in activation_type.lower(): return nn.ReLU(inplace = True)
	elif "elu" in activation_type.lower(): return nn.ELU(inplace = True)
	elif "tanh" in activation_type.lower(): return nn.Tanh()
    
def get_ConvBlock(in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, useBatchNorm = True, activation_type = "relu"):
	layers = [
		nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
	]
	if useBatchNorm: layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
	if activation_type: layers.append(get_Activation(activation_type = activation_type))
	return nn.Sequential(*layers)
    
def get_ConvTransposeBlock(in_channels, out_channels, kernel_size, stride = 1, padding = 0, output_padding = 0, dilation = 1, useBatchNorm = True, activation_type = "relu"):
	layers = [
		nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding = output_padding, dilation = dilation)
	]
	if useBatchNorm: layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
	if activation_type: layers.append(get_Activation(activation_type = activation_type))
	return nn.Sequential(*layers)

# Qiang's AutoEncoder
class Qiang_AutoEncoder(nn.Module):
	'''
	AutoEncoder Implementation by Qiang et. al based on "Anomaly Detection for Images using Auto-Encoder based Sparse Representation"
	Author's code
	'''
	def __init__(self):
		super(Qiang_AutoEncoder, self).__init__()
		# print(self)
		self.encoder = nn.Sequential(
			nn.Conv2d(Config.channels, Config.ImageSize, 4, 4, 0, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(Config.ImageSize, Config.ImageSize * 4, 4, 4, 0, bias=False),
			nn.BatchNorm2d(Config.ImageSize * 4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(Config.ImageSize * 4, Config.ImageSize * 16, 4, 4, 0, bias=False),
			nn.BatchNorm2d(Config.ImageSize * 16),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(Config.ImageSize * 16, Config.EmbeddingSize, 2, 1, 0, bias=False),
			nn.Sigmoid()
		)  
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(Config.EmbeddingSize, Config.ImageSize * 16, 2, 1, 0, bias=False),
			nn.BatchNorm2d(Config.ImageSize * 16),
			nn.ReLU(True),
			nn.ConvTranspose2d(Config.ImageSize * 16, Config.ImageSize * 4, 4, 4, 0, bias=False),
			nn.BatchNorm2d(Config.ImageSize * 4),
			nn.ReLU(True),
			nn.ConvTranspose2d(Config.ImageSize * 4, Config.ImageSize, 4, 4, 0, bias=False),
			nn.BatchNorm2d(Config.ImageSize),
			nn.ReLU(True),
			nn.ConvTranspose2d(Config.ImageSize, Config.channels, 4, 4, 0, bias=False),
			nn.Sigmoid()
		)

	def forward(self, inp):
		encoding = self.encoder(inp)
		reconstruction = self.decoder(encoding)
		return reconstruction, encoding

class AE(nn.Module):
	'''
	Modified, simple AutoEncoder
	'''
	def __init__(self, full = False):
		super(AE, self).__init__()
		# print(self)
		self.filter_count = [64, 128, 256, 512]

		self.encoder = nn.Sequential(
			get_ConvBlock(in_channels = Config.channels, out_channels = self.filter_count[0], kernel_size = 3, stride = 2, padding =  0, useBatchNorm = False),
			get_ConvBlock(in_channels = self.filter_count[0], out_channels = self.filter_count[1], kernel_size = 3, stride = 2, padding =  0, activation_type = "leaky_relu"),
			get_ConvBlock(in_channels = self.filter_count[1], out_channels = self.filter_count[2], kernel_size = 5, stride = 3, padding =  0, activation_type = "leaky_relu"),
			get_ConvBlock(in_channels = self.filter_count[2], out_channels = self.filter_count[3], kernel_size = 5, stride = 2, padding =  0, activation_type = "leaky_relu"),
			get_ConvBlock(in_channels = self.filter_count[3], out_channels = Config.EmbeddingSize, kernel_size = 3, stride = 2, padding =  0, useBatchNorm = False, activation_type = "sigmoid"),
			)

		self.decoder = nn.Sequential(
			get_ConvTransposeBlock(in_channels = Config.EmbeddingSize, out_channels = self.filter_count[3], kernel_size = 3, stride = 2, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[3], out_channels = self.filter_count[2], kernel_size = 5, stride = 2, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[2], out_channels = self.filter_count[1], kernel_size = 5, stride = 3, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[1], out_channels = self.filter_count[0], kernel_size = 5, stride = 2, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[0], out_channels = self.filter_count[0]//4, kernel_size = 5, stride = 2, padding = 0),
			get_ConvTransposeBlock(in_channels = self.filter_count[0] // 4, out_channels = Config.channels, kernel_size = 4, stride = 1, padding = 0, useBatchNorm = False, activation_type = "sigmoid")
		)

	def forward(self, inp):
		encoding = self.encoder(inp)
		reconstruction = self.decoder(encoding)
		return reconstruction, encoding

# Contractive
# Derive Base Encoder and use the same

# ACB
class ACB(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0 , dilation = 1, useBatchNorm = True, activation_type = "relu"):
		super(ACB, self).__init__()
		s_layers = [
			nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
					  stride = stride, padding = (kernel_size//2, kernel_size//2), dilation = dilation)
		]
		if useBatchNorm: s_layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
		self.s = nn.Sequential(*s_layers)

		h_layers = [
			nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (1, kernel_size),
					  stride = stride, padding = (0, kernel_size//2), dilation = dilation)
		]
		if useBatchNorm: h_layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
		self.h = nn.Sequential(*h_layers)

		v_layers = [
			nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (kernel_size, 1),
					  stride = stride, padding = (kernel_size // 2, 0), dilation = dilation)
		]
		if useBatchNorm: v_layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
		self.v = nn.Sequential(*v_layers)

		if activation_type: self.activation = get_Activation(activation_type)
		else: self.activation = lambda x: x

	def forward(self, x):
		return self.activation(self.s(x) + self.h(x) + self.v(x))

class ADB(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, output_padding = 0,  useBatchNorm = True, activation_type = "relu"):
		super(ADB, self).__init__()
		s_layers = [
			nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, output_padding = output_padding)
		]
		if useBatchNorm: s_layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
		self.s = nn.Sequential(*s_layers)

		h_layers = [
			nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (1, kernel_size), stride = stride, padding = padding, dilation = (kernel_size, 1), output_padding = (kernel_size-1, 0))
		]
		if useBatchNorm: h_layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
		self.h = nn.Sequential(*h_layers)

		v_layers = [
			nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (kernel_size, 1), stride = stride, padding = padding, dilation = (1, kernel_size), output_padding = (0, kernel_size-1))
		]
		if useBatchNorm: v_layers.append(nn.BatchNorm2d(out_channels, track_running_stats = True))
		self.v = nn.Sequential(*v_layers)

		if activation_type: self.activation = get_Activation(activation_type)
		else: self.activation = lambda x: x

	def forward(self, x):
		return self.activation(self.s(x) + self.h(x) + self.v(x))

class KS_AE(AE):
	def __init__(self, normalDecoder = False):
		super(KS_AE, self).__init__()
		# print(self)
		self.filter_count = [64, 64, 128, 256]

		self.encoder = nn.Sequential(
			ACB(in_channels = Config.channels, out_channels = self.filter_count[0], kernel_size = 3, stride = 2, padding =  0, useBatchNorm = False),
			ACB(in_channels = self.filter_count[0], out_channels = self.filter_count[1], kernel_size = 3, stride = 2, padding =  0),
			ACB(in_channels = self.filter_count[1], out_channels = self.filter_count[2], kernel_size = 5, stride = 3, padding =  0),
			ACB(in_channels = self.filter_count[2], out_channels = self.filter_count[3], kernel_size = 5, stride = 3, padding =  0),
			get_ConvBlock(in_channels = self.filter_count[3], out_channels = Config.EmbeddingSize, kernel_size = 3, stride = 2, padding =  0, useBatchNorm = False, activation_type = "sigmoid"),
			)
		
		if not normalDecoder:
			self.decoder = nn.Sequential(
				ADB(in_channels = Config.EmbeddingSize, out_channels = self.filter_count[3], kernel_size = 3, stride = 2, padding = 0),
				ADB(in_channels = self.filter_count[3], out_channels = self.filter_count[2], kernel_size = 5, stride = 2, padding = 0),
				ADB(in_channels = self.filter_count[2], out_channels = self.filter_count[1], kernel_size = 5, stride = 3, padding = 0),
				ADB(in_channels = self.filter_count[1], out_channels = self.filter_count[0], kernel_size = 5, stride = 2, padding = 0),
				ADB(in_channels = self.filter_count[0], out_channels = self.filter_count[0]//4, kernel_size = 5, stride = 2, padding = 0),
				get_ConvTransposeBlock(in_channels = self.filter_count[0]//4, out_channels = Config.channels, kernel_size = 4, stride = 1, padding = 0, useBatchNorm = False, activation_type = "sigmoid")
			)

	def forward(self, x):
		encoding = self.encoder(x)
		reconstruction = self.decoder(encoding)
		return reconstruction, encoding

# Variatonal
class VAE(AE):
	def __init__(self):
		super(VAE, self).__init__()
		# print(self)
		self.parameter_size = Config.EmbeddingSize
		self.mean_fc = nn.Linear(Config.EmbeddingSize, self.parameter_size)
		self.var_fc = nn.Linear(Config.EmbeddingSize, self.parameter_size)
	
	def encode(self, x):
		e = self.encoder(x).view(-1, Config.EmbeddingSize)
		mean = nn.ReLU()(self.mean_fc(e))
		var = nn.ReLU()(self.var_fc(e))
		return mean, var
	
	def decode(self, encoding):
		return self.decoder(encoding.view(-1, Config.EmbeddingSize, 1, 1))
	
	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar

# Variatonal
class KS_VAE(KS_AE):
	def __init__(self):
		super(KS_VAE, self).__init__()
		# print(self)
		self.parameter_size = Config.EmbeddingSize
		self.mean_fc = nn.Linear(Config.EmbeddingSize, self.parameter_size)
		self.var_fc = nn.Linear(Config.EmbeddingSize, self.parameter_size)
	
	def encode(self, x):
		e = self.encoder(x).view(-1, Config.EmbeddingSize)
		mean = nn.ReLU()(self.mean_fc(e))
		var = nn.ReLU()(self.var_fc(e))
		return mean, var
	
	def decode(self, encoding):
		return self.decoder(encoding.view(-1, Config.EmbeddingSize, 1, 1))
	
	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar

# Split
def BN_R(n, p = 0.2):
	'''
	Activation Block: ReLU(BN(x))

	Args:
		n - number of output channels as <int>
		p - probability of dropout as <float>
	Returns:
		<nn.Sequential> block
	'''
	return nn.Sequential(
		nn.BatchNorm2d(n), #, momentum = None, track_running_stats = False),
		nn.ReLU6(inplace = True), # Found ReLU6 to be performing better thatn ReLU (Improvement)
		nn.Dropout2d(p) # Slows down training but better generalization (Improvement)
	)

class Neo_AE(AE):
	def __init__(self):
		super(Neo_AE, self).__init__()

		self.encoder = nn.Sequential(
			FactorizedConv2dAdd(Config.channels, self.filter_count[0], 3, 2),
			nn.ReLU(inplace=True),
			FactorizedConv2dAdd(self.filter_count[0], self.filter_count[1], 3, 2),
			BN_R(self.filter_count[1]),
			FactorizedConv2dAdd(self.filter_count[1], self.filter_count[2], 5, 3),
			BN_R(self.filter_count[2]),
			FactorizedConv2dAdd(self.filter_count[2], self.filter_count[3], 5, 3),
			BN_R(self.filter_count[3]),
			nn.Conv2d(self.filter_count[3], Config.EmbeddingSize, 4, 3),
			nn.Sigmoid()
		)

	def forward(self, x):
		encoding = self.encoder(x)
		reconstruction = self.decoder(encoding)
		return reconstruction, encoding