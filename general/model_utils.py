from .all_imports import *

'''
Conv2D types:
    1. nn.Conv2d
    2. Conv2d_Factorized
    3. Conv2d_Depthwise
    
Conv3D types:
    1. nn.Conv3d
    2. Conv3d_Factorized
    3. Conv3d_Depthwise
'''

activations_dict = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh()
}

# ------------------------------------------------------- #
def count_parameters(model, onlyTrainable = True):
    if onlyTrainable:
        return sum([p.numel() for p in model.parameters() if p.requires_grad])
    else:
        return sum([p.numel() for p in model.parameters()])

def HalfPrecision(model, first = True):
    if first: model.half()
    
    if isinstance(model, torch.nn.modules.batchnorm._BatchNorm):
        model.float()
    for child in model.children():
        HalfPrecision(child, False)
    return model
    
def DataParallel(model):
    if torch.cuda.device_count() > 1:
#         INFO("Using %d GPUs"%(torch.cuda.device_count()))
        return nn.DataParallel(model)
    return model

# util functions
def getConvOutputShape(input_size, kernel_size, stride = 1, padding = 0):
    return ((input_size - kernel_size + (2 * padding)) // stride) + 1

def getConvTransposeOutputShape(input_size, kernel_size, stride = 1, padding = 0, output_padding = 0):
    return ((input_size - 1)*stride - (2 * padding) + kernel_size + output_padding)

def BN_A(n, activation_type = "leaky_relu", is3d = True):
    layers = list()
    if is3d: layers.append(nn.BatchNorm3d(n))
    else: layers.append(nn.BatchNorm2d(n))
    layers.append(activations_dict[activation_type])
    return nn.Sequential(*layers)

# ------------------------------------------------------- #

# Conv Elements
# C2D
class C2D_FC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1):
        super(C2D_FC, self).__init__()
        self.h = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride = (1,stride), padding=(0, padding), dilation = dilation, padding_mode = "replicate")
        self.v = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), stride = (stride,1), padding=(padding, 0), dilation = dilation, padding_mode = "replicate")

    def forward(self, x):
        return self.v(self.h(x))
    
class C2D_DS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1, dilation = 1):
        super(C2D_DS, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride = stride, padding=padding, groups=in_channels, dilation = dilation, padding_mode = "replicate")
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation = dilation, padding_mode = "replicate")

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# C3D
class C3D_FC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1):
        super(C3D_FC, self).__init__()
        self.h = nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, 1), stride = (1,stride,1), padding=(0,padding,0), dilation = dilation, padding_mode = "replicate")
        self.v = nn.Conv3d(out_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride = (stride,1,1), padding=(padding,0,0), dilation = dilation, padding_mode = "replicate")
        self.d = nn.Conv3d(out_channels, out_channels, kernel_size=(1,1,kernel_size), stride = (1,1,stride), padding=(0,0,padding), dilation = dilation, padding_mode = "replicate")

    def forward(self, x):
        return self.d(self.v(self.h(x)))
    
class C3D_DS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = (0,0,0), dilation = 1):
        super(C3D_DS, self).__init__()
        if not isinstance(kernel_size, int): kd, kw, kh = kernel_size
        else: kd, kw, kh = [kernel_size]*3
        if not isinstance(stride, int): sd, sw, sh = stride
        else: sd, sw, sh = [stride]*3
        if not isinstance(padding, int): pd, pw, ph = padding
        else: pd, pw, ph = [padding]*3
        self.main_conv = nn.Conv3d(in_channels, in_channels, (1, kw, kh), (1, sw, sh), (0, pw, ph), dilation = dilation, padding_mode = "replicate")
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, (kd, 1, 1), (sd, 1, 1), (pd, 0, 0), dilation = dilation, padding_mode = "replicate")
    
    def forward(self, x):
        return self.pointwise_conv(self.main_conv(x))

class C3D_FDS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = (0,0,0), dilation = 1):
        super(C3D_FDS, self).__init__()
        if not isinstance(kernel_size, int): kd, kw, kh = kernel_size
        else: kd, kw, kh = [kernel_size]*3
        if not isinstance(stride, int): sd, sw, sh = stride
        else: sd, sw, sh = [stride]*3
        if not isinstance(padding, int): pd, pw, ph = padding
        else: pd, pw, ph = [padding]*3
        self.width_conv = nn.Conv3d(in_channels, in_channels, (1, kw, 1), (1, sw, sh), (0, kw//2, 0), dilation = dilation, padding_mode = "replicate")
        self.height_conv = nn.Conv3d(in_channels, in_channels, (1, 1, kh), (1, sw, sh), (0, 0, kh//2), dilation = dilation, padding_mode = "replicate")
        self.depth_conv = nn.Conv3d(in_channels, out_channels, (kd, 1, 1), (sd, 1, 1), (pd, 0, 0), dilation = dilation, padding_mode = "replicate")
    
    def forward(self, x):
        return self.depth_conv(self.width_conv(x) + self.height_conv(x))

# ------------------------------------------------------- #
conv2d_dict = {
    "conv2d": nn.Conv2d,
    "conv2d_fc": C2D_FC,
#     "conv2d_acb": C2D_ACB,
    "conv2d_ds": C2D_DS
}

conv3d_dict = {
    "conv3d": nn.Conv3d,
    "conv3d_fc": C3D_FC,
#     "conv3d_acb": C3D_ACB,
    "conv3d_ds": C3D_DS,
    "conv3d_fds": C3D_FDS,
}

conv2d_transpose_dict = {
    "conv2d_transpose": nn.ConvTranspose2d
}

conv3d_transpose_dict = {
    "conv3d_transpose": nn.ConvTranspose3d
}
# ------------------------------------------------------- #

# Conv Blocks
# C2D blocks

def C2D_BN_A(in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, activation_type = "leaky_relu", conv_type = "conv2d"):
    conv_layer = conv2d_dict[conv_type]
    layers = [
        conv_layer(in_channels, out_channels, kernel_size, stride, padding = padding, dilation = dilation),
        nn.BatchNorm2d(out_channels),
    ]
    if activation_type: layers += [activations_dict[activation_type]]
        
    return nn.Sequential(*layers)

class C2D_Res(nn.Module):
    def __init__(self, channels, kernel_size, activation_type = "leaky_relu", conv_type = "conv2d"):
        super(C2D_Res, self).__init__()
        self.conv_block = C2D_BN_A(channels, channels, kernel_size, stride = 1, padding = kernel_size//2, conv_type = conv_type)
        layers = [nn.BatchNorm2d(channels)]
        if activation_type: layers += [activations_dict[activation_type]]
        self.BA = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.BA(x + self.conv_block(x))

class C2D_ACB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0 , dilation = 1, useBatchNorm = True, activation_type = "leaky_relu", conv_type = "conv2d"):
        super(C2D_ACB, self).__init__()
        conv_layer = conv2d_dict[conv_type]
        self.s = conv_layer(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                      stride = stride, padding = (kernel_size//2, kernel_size//2), dilation = dilation)
        self.h = conv_layer(in_channels = in_channels, out_channels = out_channels, kernel_size = (1, kernel_size),
                      stride = stride, padding = (0, kernel_size//2), dilation = dilation)
        self.v = conv_layer(in_channels = in_channels, out_channels = out_channels, kernel_size = (kernel_size, 1),
                      stride = stride, padding = (kernel_size // 2, 0), dilation = dilation)         
        
        layers = [nn.BatchNorm2d(out_channels)]
        if activation_type: layers += [activations_dict[activation_type]]
        self.BA = nn.Sequential(*layers)

    def forward(self, x):
        return self.BA(self.s(x) + self.h(x) + self.v(x))
    
# CT2D blocks

def CT2D_BN_A(in_channels, out_channels, kernel_size, stride = 1, padding = 0, output_padding = 0, dilation = 1, activation_type = "leaky_relu", conv_type = "conv2d_transpose"):
    conv_layer = conv2d_transpose_dict[conv_type]
    layers = [
        conv_layer(in_channels, out_channels, kernel_size, stride, padding = padding, dilation = dilation, output_padding = output_padding),
        nn.BatchNorm2d(out_channels),
    ]
    if activation_type: layers += [activations_dict[activation_type]]
        
    return nn.Sequential(*layers)

class CT2D_Res(nn.Module):
    def __init__(self, channels, kernel_size, activation_type = "leaky_relu", conv_type = "conv2d_transpose"):
        super(CT2D_Res, self).__init__()
        self.conv_t_block = CT2D_BN_A(channels, channels, kernel_size, stride = 1, padding = kernel_size//2, conv_type = conv_type)
        
        layers = [nn.BatchNorm2d(channels)]
        if activation_type: layers += [activations_dict[activation_type]]
        self.BA = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.BA(x + self.conv_t_block(x))

class CT2D_ADB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, output_padding = 0,  useBatchNorm = True, activation_type = "leaky_relu", conv_type = "conv2d_transpose"):
        super(CT2D_ADB, self).__init__()
        conv_layer = conv2d_transpose_dict[conv_type]
        
        self.s = conv_layer(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, output_padding = output_padding)
        self.h = conv_layer(in_channels = in_channels, out_channels = out_channels, kernel_size = (1, kernel_size), stride = stride, padding = padding, dilation = (kernel_size, 1), output_padding = (kernel_size-1, 0))
        self.v = conv_layer(in_channels = in_channels, out_channels = out_channels, kernel_size = (kernel_size, 1), stride = stride, padding = padding, dilation = (1, kernel_size), output_padding = (0, kernel_size-1))
        
        layers = [nn.BatchNorm2d(out_channels)]
        if activation_type: layers += [activations_dict[activation_type]]
        self.BA = nn.Sequential(*layers)

    def forward(self, x):
        return self.BA(self.s(x) + self.h(x) + self.v(x))

# C3D blocks
def C3D_BN_A(in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, activation_type = "leaky_relu", conv_type = "conv3d"):
    conv_layer = conv3d_dict[conv_type]
    layers = [
        conv_layer(in_channels, out_channels, kernel_size, stride, padding = padding, dilation = dilation),
        nn.BatchNorm3d(out_channels),
    ]
    if activation_type: layers += [activations_dict[activation_type]]
        
    return nn.Sequential(*layers)

class C3D_Res(nn.Module):
    def __init__(self, channels, kernel_size = 3, stride = 1, activation_type = "leaky_relu", conv_type = "conv3d"):
        super(C3D_Res, self).__init__()
        if not (isinstance(kernel_size, list) or isinstance(kernel_size, tuple)):
            kernel_size = [kernel_size] * 3
        self.conv_block = C3D_BN_A(channels, channels, kernel_size, stride, padding = (kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2), conv_type = conv_type)
        
        layers = [nn.BatchNorm3d(channels)]
        if activation_type: layers += [activations_dict[activation_type]]
        self.BA = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.BA(self.conv_block(x) + x)
    
# CT3D blocks
def CT3D_BN_A(in_channels, out_channels, kernel_size, stride = 1, padding = 0, output_padding = 0, dilation = 1, activation_type = "leaky_relu", conv_type = "conv3d_transpose"):
    conv_layer = conv3d_transpose_dict[conv_type]
    layers = [
        conv_layer(in_channels, out_channels, kernel_size, stride, padding = padding, dilation = dilation, output_padding = output_padding),
        nn.BatchNorm3d(out_channels),
    ]
    if activation_type: layers += [activations_dict[activation_type]]
        
    return nn.Sequential(*layers)

class CT3D_Res(nn.Module):
    def __init__(self, channels, kernel_size = 3, stride = 1, activation_type = "leaky_relu", conv_type = "conv3d_transpose"):
        super(CT3D_Res, self).__init__()
        if not (isinstance(kernel_size, list) or isinstance(kernel_size, tuple)):
            kernel_size = [kernel_size] * 3
        self.conv_t_block = CT3D_BN_A(channels, channels, kernel_size, stride, padding = (kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2), conv_type = conv_type)
        
        layers = [nn.BatchNorm3d(channels)]
        if activation_type: layers += [activations_dict[activation_type]]
        self.BA = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.BA(self.conv_t_block(x) + x)
# ---------------------------------------- # 
    
class TimeDistributed(nn.Module):
    '''
    2nd dim / dim = 1
    self.module(x[:,i,...])
    '''
    def __init__(self, module):
        # bs, ts, c, w, h
        super(TimeDistributed, self).__init__()
        self.module = module
        
    def forward(self, x):
        outputs = list()
        for i in range(x.size(1)):
            outputs += [self.module(x[:,i,...])]
        return torch.stack(outputs).transpose(0,1)
    
def moving_average_1d(window = 11):
    mean_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=window, padding = window//2)
    kernel_weights = np.array([1.0]*window)/window
    mean_conv.weight.data = torch.FloatTensor(kernel_weights).view(1, 1, -1)
    for p in mean_conv.parameters():
        p.requires_grad = False
    return mean_conv    