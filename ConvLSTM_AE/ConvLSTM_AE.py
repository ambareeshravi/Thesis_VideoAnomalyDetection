import sys
sys.path.append("..")
from general.model_utils import *
from general.all_imports import *

class ConvLSTM_Cell(nn.Module):
    def __init__(
        self,
        input_size,
        in_channels = 32,
        out_channels = 32,
        kernel_size = 3,
        stride = 1,
        padding = 0,
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
        input_size,
        in_channels = 32,
        out_channels = 32,
        kernel_size = 3,
        stride = 1,
        padding = 0,
        output_padding = 0,
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
        self.hidden_padding = self.kernel_size//2
        
        self.conv_Wh = nn.ConvTranspose2d(self.out_channels, 4*self.out_channels, self.hidden_kernel_size, 1, self.hidden_padding, output_padding = output_padding, bias = useBias)
        self.states_shape = [self.out_channels, self.output_shape, self.output_shape]
        
class CLSTM_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filter_count = [64,64,128,128],
        filter_sizes = [3,3,3,5],
        filter_strides = [2,2,2,2],
        n_lstm_layers = 2,
        useBias = False
    ):
        super(CLSTM_AE, self).__init__()
        self.__name__ = "CLSTM_AE_v2_%d"%(image_size)
        self.channels = channels
        self.image_size = image_size
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        
        self.n_layers = len(self.filter_count)
        self.n_lstm_layers = n_lstm_layers
        self.n_normal = self.n_layers - self.n_lstm_layers
        
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_lstm_layers:
                insert = TimeDistributed(C2D_BN_A(in_channels, n, k, s))
            else:
                insert = ConvLSTM_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias)
            self.encoder_layers.append(insert)
            current_input_shape = getConvOutputShape(current_input_shape, k, s)
            in_channels = n
            
        self.decoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count[::-1], self.filter_sizes[::-1], self.filter_strides[::-1])):
            oc_idx = len(self.filter_count) - (2 + idx)
            activation_type = "leaky_relu"
            if oc_idx > -1: out_channels = self.filter_count[oc_idx]
            else:
                out_channels = self.channels
                if k%2 !=0: k += 1
                activation_type = "sigmoid"
            if (self.n_layers - idx) <= self.n_lstm_layers:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            else:
                insert = ConvTransposeLSTM_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
        
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)
                                
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_lstm_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h
        
        states_list = [None] * 2 * self.n_lstm_layers
        current_input = preliminary_encodings
        lstm_outputs = list()
        
        for idx, layer in enumerate(self.encoder_layers[self.n_normal:] + self.decoder_layers[:self.n_lstm_layers]):
            layer_outputs = list()
            states = states_list[idx]
            for t in range(ts):
                states = layer(current_input[:,t,...], states)
                layer_outputs.append(states[0])
            layer_output = torch.stack(layer_outputs, dim = 1) # b,ts,c,w,h
            lstm_outputs.append(layer_output)
            current_input = layer_output
            states_list[idx] = states
        
        reconstructions = nn.Sequential(*self.decoder_layers[self.n_normal:])(layer_output)
        encodings = lstm_outputs[self.n_lstm_layers - 1].transpose(1,2)
        reconstructions = reconstructions.transpose(1,2)
        return reconstructions, encodings