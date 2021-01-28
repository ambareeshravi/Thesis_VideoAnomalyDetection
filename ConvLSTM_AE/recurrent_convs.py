import sys
sys.path.append("..")
from general.utils import most_common
from general.model_utils import *
from general.all_imports import *

from AutoEncoders.C2D_Models import *

class ConvRNN_Cell(nn.Module):
    def __init__(
        self,
        input_size:int,
        in_channels:int = 32,
        out_channels:int = 32,
        kernel_size:int = 3,
        stride:int = 1,
        padding:int = 0,
        useBias:int = True
    ):
        '''
        h_t = f(h_t-1, x_t; theta)
        
        a_t = W_ih * x + W_hh * h_t-1 + b_a
        h_t = tanh(a_t)
        o = W_oh * h_t + b_o
        y_t = softmax(o_t)
        '''
        super(ConvRNN_Cell, self).__init__()
        # Params
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.conv_Wx = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias = useBias)
        self.output_shape = getConvOutputShape(self.input_size, self.kernel_size, self.stride, self.padding)
        
        self.hidden_padding = self.kernel_size//2
        self.conv_Wh = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, 1, self.hidden_padding, bias = useBias)
        self.states_shape = [self.out_channels, self.output_shape, self.output_shape]
        
        self.conv_Wo = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, 1, self.hidden_padding, bias = useBias)
        
    def init_states(self, bs):
        return torch.autograd.Variable(torch.zeros(tuple([bs] + self.states_shape), device = self.conv_Wx.weight.device), requires_grad = True)
    
    def forward(self, x, h_p = None):
        bs, ch, w, h = x.shape
        if h_p == None: h_p = self.init_states(bs)
        h_n = self.tanh(self.conv_Wx(x) + self.conv_Wh(h_p))
        y_n = self.sigmoid(self.conv_Wo(h_n))
        return y_n, h_n
    
class ConvTransposeRNN_Cell(ConvRNN_Cell):
    def __init__(
        self,
        input_size:int,
        in_channels:int = 32,
        out_channels:int = 32,
        kernel_size:int = 3,
        stride:int = 1,
        padding:int = 0,
        output_padding:int = 0,
        useBias = True
    ):
        super(ConvTransposeRNN_Cell, self).__init__(input_size = input_size)
        # Params
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv_Wx = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, output_padding = output_padding, bias = useBias)
        self.output_shape = getConvTransposeOutputShape(self.input_size, self.kernel_size, self.stride, self.padding, output_padding)
        
        self.hidden_kernel_size = (self.kernel_size - 1) if (self.kernel_size%2==0) else self.kernel_size
        self.hidden_padding = (self.kernel_size - 1)//2
        
        self.conv_Wh = nn.ConvTranspose2d(self.out_channels, self.out_channels, self.hidden_kernel_size, 1, self.hidden_padding, output_padding = output_padding, bias = useBias)
        self.states_shape = [self.out_channels, self.output_shape, self.output_shape]
        
        self.conv_Wo = nn.ConvTranspose2d(self.out_channels, self.out_channels, self.hidden_kernel_size, 1, self.hidden_padding, output_padding = output_padding, bias = useBias)
        
class ConvLSTM_Cell(nn.Module):
    def __init__(
        self,
        input_size:int,
        in_channels:int = 32,
        out_channels:int = 32,
        kernel_size:int = 3,
        stride:int = 1,
        padding:int = 0,
        useBias = True
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
        super(ConvLSTM_Cell, self).__init__()
        # Params
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.conv_Wx = nn.Conv2d(self.in_channels, 4*self.out_channels, self.kernel_size, self.stride, self.padding, bias = useBias)
        self.output_shape = getConvOutputShape(self.input_size, self.kernel_size, self.stride, self.padding)
        
        self.hidden_padding = self.kernel_size//2
        self.conv_Wh = nn.Conv2d(self.out_channels, 4*self.out_channels, self.kernel_size, 1, self.hidden_padding, bias = useBias)
        self.states_shape = [self.out_channels, self.output_shape, self.output_shape]
        
    def init_states(self, bs):
        return (
            torch.autograd.Variable(torch.zeros(tuple([bs] + self.states_shape), device = self.conv_Wx.weight.device), requires_grad = True),
            torch.autograd.Variable(torch.zeros(tuple([bs] + self.states_shape), device = self.conv_Wx.weight.device), requires_grad = True)
        )
    
    def forward(self, x, p_states = None):
        bs, ch, w, h = x.shape
        if p_states == None: p_states = self.init_states(bs)
        h_p, c_p = p_states
        
        conv_out = self.conv_Wx(x) + self.conv_Wh(h_p)
        
        i, f, c, o = conv_out.split(self.out_channels, dim = 1)
        
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        c_n = (f * c_p) + (i * self.tanh(c))
        o = self.sigmoid(o)
        h_n = o * self.tanh(c_n)
        return h_n, c_n
    
class ConvTransposeLSTM_Cell(ConvLSTM_Cell):
    def __init__(
        self,
        input_size:int,
        in_channels:int = 32,
        out_channels:int = 32,
        kernel_size:int = 3,
        stride:int = 1,
        padding:int = 0,
        output_padding:int = 0,
        useBias = True
    ):
        super(ConvTransposeLSTM_Cell, self).__init__(input_size = input_size)
        # Params
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv_Wx = nn.ConvTranspose2d(self.in_channels, 4*self.out_channels, self.kernel_size, self.stride, self.padding, output_padding = output_padding, bias = useBias)
        self.output_shape = getConvTransposeOutputShape(self.input_size, self.kernel_size, self.stride, self.padding, output_padding)
        
        self.hidden_kernel_size = (self.kernel_size - 1) if (self.kernel_size%2==0) else self.kernel_size
        self.hidden_padding = (self.kernel_size - 1)//2
        
        self.conv_Wh = nn.ConvTranspose2d(self.out_channels, 4*self.out_channels, self.hidden_kernel_size, 1, self.hidden_padding, output_padding = output_padding, bias = useBias)
        self.states_shape = [self.out_channels, self.output_shape, self.output_shape]