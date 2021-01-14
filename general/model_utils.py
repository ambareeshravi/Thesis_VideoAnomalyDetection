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

class Conv2dLSTM_Cell(nn.Module):
    def __init__(self, 
                 image_size,
                 input_dim = 32,
                 hidden_dim = 32,
                 kernel_size = 3,
                 stride = 1,
                 padding = 0,
                 include_W = False,
                 conv_bias = False,
                 conv_type = "conv2d",
                 init_random = False,
                ):
        '''
        Using channels first n,c,w,h for images
                    
        i = sigma(W_xi * X_t + W_hi * H_t-1 + W_ci.C_t-1 + b_i)
        f_t = sigma(W_xf * X_t + W_hf * H_t-1 + W_cf.C_t-1 + b_f)
        C_t = f_t.C_t-1 + i_t.tanh(W_xc*X_t + W_hc*H_t-1 + b_c)
        o_t = sigma(W_xo * X_t + W_ho * H_t-1 + W_co.C_t + b_o)
        H_t = o_t.tanh(C_t) 
        
        return_sequence - True:
            (samples, time_steps, filters, rows, cols)
        return_sequence - False:
            (samples, filters, rows, cols)

        '''
        super(Conv2dLSTM_Cell, self).__init__()
#         self.device = torch.device("cpu")
#         if useGPU and torch.cuda.is_available: self.device = torch.device("cuda")
        self.image_size = image_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_random = init_random
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.include_W = include_W
        self.conv_bias = conv_bias
        
        self.output_shape = getConvOutputShape(self.image_size, self.kernel_size, self.stride, self.padding)
        
        isACB = False
        self.conv_layer = nn.Conv2d
        inp_conv_kwargs = {
            "in_channels": self.input_dim,
            "out_channels": self.hidden_dim * 4,
            "kernel_size": self.kernel_size,
            "stride": self.stride, 
            "padding": self.padding,
        }
        
        if "acb" in conv_type.lower():
            isACB = True
            self.conv_layer = C2D_ACB
            inp_conv_kwargs["useBatchNorm"] = False
            inp_conv_kwargs["activation_type"] = False
            
        self.hidden_kernel_size = (self.kernel_size - 1) if (self.kernel_size%2==0) else self.kernel_size
        
        self.conv_output = (
            getConvOutputShape(self.image_size, self.kernel_size, self.stride, padding = self.kernel_size//2 if isACB else self.padding),
            getConvOutputShape(self.image_size, self.kernel_size, self.stride, padding = self.kernel_size//2 if isACB else self.padding) 
                           )
        
        hid_conv_kwargs = {
            "in_channels": self.hidden_dim,
            "out_channels": self.hidden_dim * 4,
            "kernel_size": self.hidden_kernel_size,
            "stride": 1, 
            "padding": self.hidden_kernel_size // 2,
            "bias": self.conv_bias
        }
        
        self.conv_Wx = self.conv_layer(**inp_conv_kwargs)
        self.conv_Wh = nn.Conv2d(**hid_conv_kwargs)
        
        if self.include_W:
            self.weights_shape = tuple([3, 1, self.hidden_dim] + list(self.conv_output))
            self.W = torch.autograd.Variable(torch.randn(*self.weights_shape), requires_grad = True)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
                
    def init_states(self, batch_size):
        state_shape = [batch_size, self.hidden_dim] + list(self.conv_output)
        init_fn = torch.zeros
        if self.init_random:
            init_fn = torch.randn
            
        return (
            init_fn(*state_shape, device = self.conv_Wx.weight.device),
            init_fn(*state_shape, device = self.conv_Wx.weight.device)
        )
            
    def forward(self, x, states = None):
        b,c,w,h = x.shape
        if states == None: states = self.init_states(b)
            
        h, c = states
        # i,f,c,o
        conv_x_out = self.conv_Wx(x)
        conv_h_out = self.conv_Wh(h)
        if self.include_W: (Wci, Wcf, Wco) = self.W.to(self.conv_Wx.weight.device) * c
        
        conv_Wxi, conv_Wxf, conv_Wxc, conv_Wxo = conv_x_out.split(self.hidden_dim, dim = 1)
        conv_Whi, conv_Whf, conv_Whc, conv_Who = conv_h_out.split(self.hidden_dim, dim = 1)
        
        i = conv_Wxi + conv_Whi
        if self.include_W: i += Wci
        i = self.sigmoid(i)
        
        f = conv_Wxf + conv_Whf
        if self.include_W: f += Wcf
        f = self.sigmoid(f)
        
        c_n = f * c + i * self.tanh(conv_Wxc + conv_Whc)
        
        o = conv_Wxo + conv_Who
        if self.include_W: o += Wco
            
        o = self.sigmoid(o)
        h_n = o * self.tanh(c_n) 
        return h_n, c_n

class ConvTranspose2dLSTM_Cell(nn.Module):
    def __init__(self, 
                 image_size,
                 input_dim = 32,
                 hidden_dim = 32,
                 kernel_size = 3,
                 stride = 1,
                 padding = 0,
                 output_padding = 0,
                 include_W = False,
                 conv_bias = False,
                 conv_type = "conv2d_transpose",
                 init_random = False,
                ):
        '''
        Using channels first n,c,w,h for images
                    
        i = sigma(W_xi * X_t + W_hi * H_t-1 + W_ci.C_t-1 + b_i)
        f_t = sigma(W_xf * X_t + W_hf * H_t-1 + W_cf.C_t-1 + b_f)
        C_t = f_t.C_t-1 + i_t.tanh(W_xc*X_t + W_hc*H_t-1 + b_c)
        o_t = sigma(W_xo * X_t + W_ho * H_t-1 + W_co.C_t + b_o)
        H_t = o_t.tanh(C_t) 
        
        return_sequence - True:
            (samples, time_steps, filters, rows, cols)
        return_sequence - False:
            (samples, filters, rows, cols)
        '''
        super(ConvTranspose2dLSTM_Cell, self).__init__()
#         self.device = torch.device("cpu")
#         if useGPU and torch.cuda.is_available: self.device = torch.device("cuda")
        self.image_size = image_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_random = init_random
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.include_W = include_W
        self.conv_bias = conv_bias
        
        self.output_shape = getConvTransposeOutputShape(self.image_size, self.kernel_size, self.stride, self.padding, self.output_padding)
        
        isADB = False
        self.conv_layer = nn.ConvTranspose2d
        inp_conv_kwargs = {
            "in_channels": self.input_dim,
            "out_channels": self.hidden_dim * 4,
            "kernel_size": self.kernel_size,
            "stride": self.stride, 
            "padding": self.padding,
            "output_padding": self.output_padding
        }
        
        if "adb" in conv_type.lower():
            isADB = True
            self.conv_layer = CT2D_ADB
            inp_conv_kwargs["useBatchNorm"] = False
            inp_conv_kwargs["activation_type"] = False
            
        self.hidden_kernel_size = (self.kernel_size - 1) if (self.kernel_size%2==0) else self.kernel_size
        # self.kernel_size//2 if isADB else self.padding
        self.conv_output = (
            getConvTransposeOutputShape(self.image_size, self.kernel_size, self.stride, padding = self.padding, output_padding = self.output_padding),
            getConvTransposeOutputShape(self.image_size, self.kernel_size, self.stride, padding = self.padding, output_padding = self.output_padding)
        )
        
        hid_conv_kwargs = {
            "in_channels": self.hidden_dim,
            "out_channels": self.hidden_dim * 4,
            "kernel_size": self.hidden_kernel_size,
            "stride": 1, 
            "padding": self.hidden_kernel_size // 2,
            "output_padding": self.output_padding,
            "bias": self.conv_bias
        }
        self.conv_Wx = self.conv_layer(**inp_conv_kwargs)
        self.conv_Wh = nn.ConvTranspose2d(**hid_conv_kwargs)
        
        if self.include_W:
            self.weights_shape = tuple([3, 1, self.hidden_dim] + list(self.conv_output))
            self.W = torch.autograd.Variable(torch.randn(*self.weights_shape), requires_grad = True)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def init_states(self, batch_size):
        state_shape = [batch_size, self.hidden_dim] + list(self.conv_output)
        init_fn = torch.zeros
        if self.init_random:
            init_fn = torch.randn
            
        return (
            init_fn(*state_shape, device = self.conv_Wx.weight.device),
            init_fn(*state_shape, device = self.conv_Wx.weight.device)
        )
            
    def forward(self, x, states = None):
        b,c,w,h = x.shape
        if states == None: states = self.init_states(b)
            
        h, c = states
        # i,f,c,o
        conv_x_out = self.conv_Wx(x)
        conv_h_out = self.conv_Wh(h)
        if self.include_W: (Wci, Wcf, Wco) = self.W.to(self.conv_Wx.weight.device) * c
        
        conv_Wxi, conv_Wxf, conv_Wxc, conv_Wxo = conv_x_out.split(self.hidden_dim, dim = 1)
        conv_Whi, conv_Whf, conv_Whc, conv_Who = conv_h_out.split(self.hidden_dim, dim = 1)
        
        i = conv_Wxi + conv_Whi
        if self.include_W: i += Wci
        i = self.sigmoid(i)
        
        f = conv_Wxf + conv_Whf
        if self.include_W: f += Wcf
        f = self.sigmoid(f)
        
        c_n = f * c + i * self.tanh(conv_Wxc + conv_Whc)
        
        o = conv_Wxo + conv_Who
        if self.include_W: o += Wco
            
        o = self.sigmoid(o)
        h_n = o * self.tanh(c_n) 
        return h_n, c_n

class Conv2dRNN_Cell(nn.Module):
    def __init__(
        self,
        image_size:int,
        input_dim:int = 32,
        hidden_dim:int = 32,
        kernel_size:int = 3,
        stride:int = 1,
        padding:int = 0,
        conv_bias:bool = False,
        conv_type:str = "conv2d",
        init_random:bool = False,
    ):
        '''
        h_t = f(h_t-1, x_t; theta)
        
        a_t = W_ih * x + W_hh * h_t-1 + b_a
        h_t = tanh(a_t)
        o = W_oh * h_t + b_o
        y_t = softmax(o_t)
        '''
        super(Conv2dRNN_Cell, self).__init__()
        self.__name__ = "Conv2dRNN_Cell"
        self.image_size = image_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_random = init_random
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_bias = conv_bias
        
        self.output_shape = getConvOutputShape(self.image_size, self.kernel_size, self.stride, self.padding)
        
        isACB = False
        self.conv_layer = nn.Conv2d
        inp_conv_kwargs = {
            "in_channels": self.input_dim,
            "out_channels": self.hidden_dim,
            "kernel_size": self.kernel_size,
            "stride": self.stride, 
            "padding": self.padding,
        }
        
        if "acb" in conv_type.lower():
            isACB = True
            self.conv_layer = C2D_ACB
            inp_conv_kwargs["useBatchNorm"] = False
            inp_conv_kwargs["activation_type"] = False
            
        self.hidden_kernel_size = (self.kernel_size - 1) if (self.kernel_size%2==0) else self.kernel_size
        
        self.conv_output = (
            getConvOutputShape(self.image_size, self.kernel_size, self.stride, padding = self.kernel_size//2 if isACB else self.padding),
            getConvOutputShape(self.image_size, self.kernel_size, self.stride, padding = self.kernel_size//2 if isACB else self.padding) 
                           )
        
        hid_conv_kwargs = {
            "in_channels": self.hidden_dim,
            "out_channels": self.hidden_dim,
            "kernel_size": self.hidden_kernel_size,
            "stride": 1, 
            "padding": self.hidden_kernel_size // 2,
            "bias": self.conv_bias
        }
        
        self.conv_Wx = self.conv_layer(**inp_conv_kwargs)
        self.conv_Wh = nn.Conv2d(**hid_conv_kwargs)
        
        self.weights_shape = tuple([1, self.hidden_dim] + list(self.conv_output))
        self.W_o = torch.autograd.Variable(torch.randn(*self.weights_shape), requires_grad = True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
                
    def init_states(self, batch_size):
        state_shape = [batch_size, self.hidden_dim] + list(self.conv_output)
        init_fn = torch.zeros
        if self.init_random:
            init_fn = torch.randn
            
        return init_fn(*state_shape, device = self.conv_Wx.weight.device)
        
    def forward(self, x, h_p = None):
        '''
        
        h_t = f(h_t-1, x_t; theta)
        
        a_t = W_ih * x + W_hh * h_t-1 + b_a
        h_t = tanh(a_t)
        o = W_oh * h_t + b_o
        y_t = softmax(o_t)
        
        '''
        b,c,w,h = x.shape
        if h_p == None: h_p = self.init_states(b)
                
        h_n = self.tanh(self.conv_Wx(x) + self.conv_Wh(h_p))
        y_n = self.sigmoid(self.W_o.to(self.conv_Wx.weight.device) * h_n)
        return y_n, h_n
    
class ConvTranspose2dRNN_Cell(nn.Module):
    def __init__(
        self,
        image_size:int,
        input_dim:int = 32,
        hidden_dim:int = 32,
        kernel_size:int = 3,
        stride:int = 1,
        padding:int = 0,
        output_padding = 0,
        conv_bias:bool = False,
        conv_type = "conv2d_transpose",
        init_random:bool = False,
    ):
        '''
        h_t = f(h_t-1, x_t; theta)
        
        a_t = W_ih * x + W_hh * h_t-1 + b_a
        h_t = tanh(a_t)
        o = W_oh * h_t + b_o
        y_t = softmax(o_t)
        '''
        super(ConvTranspose2dRNN_Cell, self).__init__()
        self.image_size = image_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_random = init_random
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.conv_bias = conv_bias
        
        self.output_shape = getConvTransposeOutputShape(self.image_size, self.kernel_size, self.stride, self.padding, self.output_padding)
        
        isADB = False
        self.conv_layer = nn.ConvTranspose2d
        inp_conv_kwargs = {
            "in_channels": self.input_dim,
            "out_channels": self.hidden_dim,
            "kernel_size": self.kernel_size,
            "stride": self.stride, 
            "padding": self.padding,
            "output_padding": self.output_padding
        }
        
        if "adb" in conv_type.lower():
            isADB = True
            self.conv_layer = CT2D_ADB
            inp_conv_kwargs["useBatchNorm"] = False
            inp_conv_kwargs["activation_type"] = False
            
        self.hidden_kernel_size = (self.kernel_size - 1) if (self.kernel_size%2==0) else self.kernel_size
        # self.kernel_size//2 if isADB else self.padding
        self.conv_output = (
            getConvTransposeOutputShape(self.image_size, self.kernel_size, self.stride, padding = self.padding, output_padding = self.output_padding),
            getConvTransposeOutputShape(self.image_size, self.kernel_size, self.stride, padding = self.padding, output_padding = self.output_padding)
        )
        
        hid_conv_kwargs = {
            "in_channels": self.hidden_dim,
            "out_channels": self.hidden_dim,
            "kernel_size": self.hidden_kernel_size,
            "stride": 1, 
            "padding": self.hidden_kernel_size // 2,
            "output_padding": self.output_padding,
            "bias": self.conv_bias
        }
        self.conv_Wx = self.conv_layer(**inp_conv_kwargs)
        self.conv_Wh = nn.ConvTranspose2d(**hid_conv_kwargs)

        self.weights_shape = tuple([1, self.hidden_dim] + list(self.conv_output))
        self.W_o = torch.autograd.Variable(torch.randn(*self.weights_shape), requires_grad = True)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def init_states(self, batch_size):
        state_shape = [batch_size, self.hidden_dim] + list(self.conv_output)
        init_fn = torch.zeros
        if self.init_random:
            init_fn = torch.randn
            
        return init_fn(*state_shape, device = self.conv_Wx.weight.device)
            
    def forward(self, x, h_p = None):
        b,c,w,h = x.shape
        if h_p == None: h_p = self.init_states(b)
        
        h_n = self.tanh(self.conv_Wx(x) + self.conv_Wh(h_p))
        y_n = self.sigmoid(self.W_o.to(self.conv_Wx.weight.device) * h_n)
        return y_n, h_n
    
class TimeDistributed(nn.Module):
    def __init__(self, module):
        # bs, ts, c, w, h
        super(TimeDistributed, self).__init__()
        self.module = module
        
    def forward(self, x):
        outputs = list()
        for i in range(x.size(1)):
            outputs += [self.module(x[:,i,...])]
        return torch.stack(outputs).transpose(0,1)
    
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)