import sys
sys.path.append("..")

from general import *
from general.all_imports import *
from general.model_utils import *
from general.losses import max_norm

# popular modules
class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk = 1, dv = 1, Nh = 1, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        if self.dk == 1: self.dk = self.out_channels // 8
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x
    
def get_AAC_BNA(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 0, conv_type = None, activation_type="leaky_relu"):
        return nn.Sequential(
            AugmentedConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            BN_A(out_channels, activation_type = activation_type, is3d=False)
        )    

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
    
# Custom Attention Modules
class LinearAttentionLayer(nn.Module):
    '''
    ut = tanh(W ht + b)
    alpha = softmax(v.T ut)
    
    s = sum(t = 1 -> M) alpha_t h_t
    '''
    def __init__(
        self,
        inp_dim
    ):
        super(LinearAttentionLayer, self).__init__()
        self.inp_dim = inp_dim
        self.Wb = nn.Linear(self.inp_dim, self.inp_dim)
        self.v = torch.autograd.Variable(torch.rand(self.inp_dim, 1), requires_grad = True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x):
        Wx = self.Wb(x)
        v_t_repeat = (self.v.T.to(self.Wb.weight.device)).repeat(Wx.shape[0], 1)
        u_t = torch.multiply(v_t_repeat, self.tanh(Wx))
        alpha = self.softmax(u_t)
        return torch.multiply(x, alpha)

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
    # Good
    def __init__(self, model, kernel_sizes = (3,5), projection = 64, channels = None, lambda_ = 1e-6, max_norm_clip = 1):
        super(ConvAttentionWapper, self).__init__()
        self.model = model
        self.projection = projection
        self.kernel_sizes = kernel_sizes
        self.lambda_ = lambda_
        self.max_norm_clip = max_norm_clip
        self.out_channels = self.model.channels
        if channels != None: self.out_channels = channels
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.model.channels, self.projection, self.kernel_sizes[0], 1, padding = self.kernel_sizes[0]//2),
            nn.Conv2d(self.projection, self.out_channels, self.kernel_sizes[1], 1, padding = self.kernel_sizes[1]//2),
        )
        self.__name__ = self.model.__name__ + "_CONV_ATTENTION_P%s_L%s_C%s"%(self.projection, self.lambda_, self.out_channels)
        self.act_block = nn.Sequential(
            nn.BatchNorm2d(self.model.channels),
            nn.Sigmoid()
        )
    
    def attention_forward(self, x):
#         x_a = self.attention_conv(x)
#         return self.act_block(x_a)
        x_a = self.attention_conv(x)
        self.xam = x_a
        return self.act_block(torch.multiply(x, x_a))
    
    def attention_loss(self, w):
        return self.lambda_ * torch.sqrt(torch.sum(w**2))
#         return torch.sum(max_norm(self.attention_conv[0].weight.data, self.max_norm_clip)) + torch.sum(max_norm(self.attention_conv[1].weight.data, self.max_norm_clip))
           
    def forward(self, x, returnMask = False):
        x_a = self.attention_forward(x)
        model_returns = list(self.model(x_a))
        if returnMask: return tuple(model_returns + [x_a, self.xam])
        return tuple(model_returns + [x_a])

class SoftMaxConvAttentionWrapper(nn.Module):
    # Very Good
    # V1 -> Summation and Tensor Multiplication
    def __init__(self, model, kernel_sizes = (3,5), projection = 64, channels = None):
        super(SoftMaxConvAttentionWrapper, self).__init__()
        self.model = model
        self.projection = projection
        self.kernel_sizes = kernel_sizes
        self.out_channels = self.model.channels
        if channels != None: self.out_channels = channels
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.model.channels, self.projection, self.kernel_sizes[0], 1, padding = self.kernel_sizes[0]//2),
            nn.Conv2d(self.projection, self.out_channels, self.kernel_sizes[1], 1, padding = self.kernel_sizes[1]//2),
            nn.BatchNorm2d(self.out_channels),
#             nn.Sigmoid()
        )
        self.__name__ = self.model.__name__ + "_SOFTMAX_ATTENTION_P%s_C%s"%(self.projection, self.out_channels)
        
    def attention_forward(self, x):
        attention_activations = self.attention_conv(x)
        v = torch.sum(attention_activations, keepdim = True, axis = -2)
        h = torch.sum(attention_activations, keepdim = True, axis = -1)
        vs = F.softmax(v, dim = -1)
        hs = F.softmax(h, dim = -2)
        x_a = torch.matmul(hs, vs)
        self.xam = x_a
        return torch.multiply(x, x_a)
           
    def forward(self, x, returnMask = False):
        x_a = self.attention_forward(x)
        model_returns = list(self.model(x_a))
        if returnMask: return tuple(model_returns + [x_a, self.xam])
        return tuple(model_returns + [x_a])
    
class SoftMaxConvAttentionWrapper2(nn.Module):
    # Okay
    # V2 -> Summation and Tensor Multiplication with normalization
    def __init__(self, model, kernel_sizes = (3,5), projection = 64, channels = None):
        super(SoftMaxConvAttentionWrapper2, self).__init__()
        self.model = model
        self.projection = projection
        self.kernel_sizes = kernel_sizes
        self.out_channels = self.model.channels
        if channels != None: self.out_channels = channels
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.model.channels, self.projection, self.kernel_sizes[0], 1, padding = self.kernel_sizes[0]//2),
            nn.Conv2d(self.projection, self.out_channels, self.kernel_sizes[1], 1, padding = self.kernel_sizes[1]//2),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )
        self.__name__ = self.model.__name__ + "_SOFTMAX_ATTENTION_2_P%s_C%s"%(self.projection, self.out_channels)
        self.max_pool = nn.MaxPool2d((5,5), 1, padding = 2)
    
    def attention_forward(self, x):
        attention_activations = self.attention_conv(x)
        v = torch.sum(attention_activations, keepdim = True, axis = -2)
        h = torch.sum(attention_activations, keepdim = True, axis = -1)
        vs = F.softmax(v, dim = -1)
        hs = F.softmax(h, dim = -2)
        x_a = torch.matmul(hs, vs)
        x_a = (x_a - x_a.min()) / (x_a.max() - x_a.min())
        self.xam = x_a
        return torch.multiply(x, x_a)
           
    def forward(self, x, returnMask = False):
        x_a = self.attention_forward(x)
        model_returns = list(self.model(x_a))
        if returnMask: return tuple(model_returns + [x_a, self.xam])
        return tuple(model_returns + [x_a])
    
class SoftMaxConvAttentionWrapper3(nn.Module):
    # Good
    # V3 -> Mean and Tensor Multiplication
    def __init__(self, model, kernel_sizes = (3,5), projection = 64, channels = None):
        super(SoftMaxConvAttentionWrapper3, self).__init__()
        self.model = model
        self.projection = projection
        self.kernel_sizes = kernel_sizes
        self.out_channels = self.model.channels
        if channels != None: self.out_channels = channels
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.model.channels, self.projection, self.kernel_sizes[0], 1, padding = self.kernel_sizes[0]//2),
            nn.Conv2d(self.projection, self.out_channels, self.kernel_sizes[1], 1, padding = self.kernel_sizes[1]//2),
            nn.BatchNorm2d(self.out_channels),
#             nn.Sigmoid()
        )
        self.__name__ = self.model.__name__ + "_SOFTMAX_ATTENTION_3_P%s_C%s"%(self.projection, self.out_channels)
        
    def attention_forward(self, x):
        attention_activations = self.attention_conv(x)
        v = torch.mean(attention_activations, keepdim = True, axis = -2)
        h = torch.mean(attention_activations, keepdim = True, axis = -1)
        vs = F.softmax(v, dim = -1)
        hs = F.softmax(h, dim = -2)
        x_a = torch.matmul(hs, vs)
        self.xam = x_a
        return torch.multiply(x, x_a)
           
    def forward(self, x, returnMask = False):
        x_a = self.attention_forward(x)
        model_returns = list(self.model(x_a))
        if returnMask: return tuple(model_returns + [x_a, self.xam])
        return tuple(model_returns + [x_a])

class SoftMaxConvAttentionWrapper4(nn.Module):
    # Poor
    # V4 -> Summation, normalozation and Tensor Multiplication
    def __init__(self, model, kernel_sizes = (3,5), projection = 64, channels = None):
        super(SoftMaxConvAttentionWrapper4, self).__init__()
        self.model = model
        self.projection = projection
        self.kernel_sizes = kernel_sizes
        self.out_channels = self.model.channels
        if channels != None: self.out_channels = channels
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.model.channels, self.projection, self.kernel_sizes[0], 1, padding = self.kernel_sizes[0]//2),
            nn.Conv2d(self.projection, self.out_channels, self.kernel_sizes[1], 1, padding = self.kernel_sizes[1]//2),
            nn.BatchNorm2d(self.out_channels),
#             nn.Sigmoid()
        )
        self.__name__ = self.model.__name__ + "_SOFTMAX_ATTENTION4_P%s_C%s"%(self.projection, self.out_channels)
        
    def normalize(self, x, axis):
        x -= x.min(axis, keepdim = True)[0]
        x /= x.max(axis, keepdim = True)[0]
        return x
    
    def attention_forward(self, x):
        attention_activations = self.attention_conv(x)
        v = torch.sum(attention_activations, keepdim = True, axis = -2)
        h = torch.sum(attention_activations, keepdim = True, axis = -1)
        vs = F.softmax(v, dim = -1)
        hs = F.softmax(h, dim = -2)
        # Normalization to improve visualization
        vs = self.normalize(vs, axis = -1)
        hs = self.normalize(hs, axis = -2)
        x_a = torch.matmul(hs, vs)
        self.xam = x_a
        return torch.multiply(x, x_a)
           
    def forward(self, x, returnMask = False):
        x_a = self.attention_forward(x)
        model_returns = list(self.model(x_a))
        if returnMask: return tuple(model_returns + [x_a, self.xam])
        return tuple(model_returns + [x_a])

class SoftMaxConvAttentionWrapper5(nn.Module):
    # V5 -> Summation, normalization, scaling, Tensor Multiplication
    def __init__(self, model, kernel_sizes = (3,5), projection = 64, channels = None, layerWrapper = False):
        super(SoftMaxConvAttentionWrapper5, self).__init__()
        self.model = model
        self.projection = projection
        self.kernel_sizes = kernel_sizes
        self.out_channels = self.model.channels
        self.layerWrapper = layerWrapper
        if channels != None: self.out_channels = channels
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.model.channels, self.projection, self.kernel_sizes[0], 1, padding = self.kernel_sizes[0]//2),
            nn.Conv2d(self.projection, self.out_channels, self.kernel_sizes[1], 1, padding = self.kernel_sizes[1]//2),
            nn.BatchNorm2d(self.out_channels),
            nn.AvgPool2d(9, stride = 1, padding = 4)
        )
        
        self.ma_conv = TimeDistributed(moving_average_1d(window = 9))
        self.__name__ = self.model.__name__ + "_SOFTMAX_ATTENTION5_P%s_C%s"%(self.projection, self.out_channels)
        
    def normalize(self, x, axis):
        x -= x.min(axis, keepdim = True)[0]
        x /= x.max(axis, keepdim = True)[0]
        return x
    
    def attention_forward(self, x):
        attention_activations = self.attention_conv(x)
        # Axis-wise sumamtion
        v = torch.mean(attention_activations, keepdim = True, axis = -2)
        h = torch.mean(attention_activations, keepdim = True, axis = -1)
        # Axis-wise Softmax
        vs = F.softmax(v, dim = -1)
        hs = F.softmax(h, dim = -2)
        # Axis-wise normalization
#         vs = self.normalize(vs, axis = -1)
#         hs = self.normalize(hs, axis = -2)
        # Axis-wise scaling
        vs = scale(vs, t_min = 3e-1)
        hs = scale(hs, t_min = 3e-1)
        # Applying moving average
        vs = self.ma_conv(vs)
        hs = self.ma_conv(hs.transpose(-1,-2)).transpose(-1,-2)
        # Axis-wise scaling
        vs = scale(vs, t_min = 3e-1)
        hs = scale(hs, t_min = 3e-1)
        # Tensor Multiplication
        x_a = torch.matmul(hs, vs)
        self.xam = x_a
        # Applying Hadamard product
        return torch.multiply(x, x_a)
        
    def forward(self, x, returnMask = False):
        x_a = self.attention_forward(x)
        model_returns = self.model(x_a)
        if self.layerWrapper: return model_returns
        model_returns = [model_returns]
        if returnMask: return tuple(model_returns + [x_a, self.xam])
        return tuple(model_returns + [x_a])

class SoftMaxConvAttentionWrapper6(nn.Module):
    # V5 -> Summation, normalization, scaling, Tensor Multiplication, pooling
    def __init__(self, model, kernel_sizes = (3,5), projection = 64, channels = None):
        super(SoftMaxConvAttentionWrapper6, self).__init__()
        self.model = model
        self.projection = projection
        self.kernel_sizes = kernel_sizes
        self.out_channels = self.model.channels
        if channels != None: self.out_channels = channels
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.model.channels, self.projection, self.kernel_sizes[0], 1, padding = self.kernel_sizes[0]//2),
            nn.Conv2d(self.projection, self.out_channels, self.kernel_sizes[1], 1, padding = self.kernel_sizes[1]//2),
            nn.BatchNorm2d(self.out_channels),
#             nn.Sigmoid()
        )
        self.max_pool = nn.MaxPool2d(9, stride = 1, padding = 4)
        self.__name__ = self.model.__name__ + "_SOFTMAX_ATTENTION6_P%s_C%s"%(self.projection, self.out_channels)
        
    def normalize(self, x, axis):
        x -= x.min(axis, keepdim = True)[0]
        x /= x.max(axis, keepdim = True)[0]
        return x
    
    def attention_forward(self, x):
        attention_activations = self.attention_conv(x)
        v = torch.sum(attention_activations, keepdim = True, axis = -2)
        h = torch.sum(attention_activations, keepdim = True, axis = -1)
        vs = self.normalize(v, axis = -1)
        hs = self.normalize(h, axis = -2)

        # Normalization to improve visualization
        vs = scale(vs, t_min = 3e-1)
        hs = scale(hs, t_min = 3e-1)
        x_a = torch.matmul(hs, vs)
        x_a = self.max_pool(x_a)
        self.xam = x_a
        return torch.multiply(x, x_a)
        
    def forward(self, x, returnMask = False):
        x_a = self.attention_forward(x)
        model_returns = list(self.model(x_a))
        if returnMask: return tuple(model_returns + [x_a, self.xam])
        return tuple(model_returns + [x_a])

class SoftMaxConvAttentionRNNWrapper(nn.Module):
    # Based on SoftMaxConvAttentionWrapper5
    # V5 -> Summation, normalization, scaling, Tensor Multiplication
    def __init__(self, channels, kernel_sizes = (3,5), projection = 64):
        super(SoftMaxConvAttentionRNNWrapper, self).__init__()
        self.channels = channels
        self.projection = projection
        self.kernel_sizes = kernel_sizes
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.channels, self.projection, self.kernel_sizes[0], 1, padding = self.kernel_sizes[0]//2),
            nn.Conv2d(self.projection, self.channels, self.kernel_sizes[1], 1, padding = self.kernel_sizes[1]//2),
            nn.BatchNorm2d(self.channels),
            nn.AvgPool2d(9, stride = 1, padding = 4)
        )
        
        self.ma_conv = TimeDistributed(moving_average_1d(window = 9))
        self.__name__ = "_SMCARnnW_P%s_C%s"%(self.projection, self.channels)
        
    def normalize(self, x, axis):
        x -= x.min(axis, keepdim = True)[0]
        x /= x.max(axis, keepdim = True)[0]
        return x
    
    def attention_forward(self, x):
        attention_activations = self.attention_conv(x)
        # Axis-wise sumamtion
        v = torch.mean(attention_activations, keepdim = True, axis = -2)
        h = torch.mean(attention_activations, keepdim = True, axis = -1)
        # Axis-wise Softmax
        vs = F.softmax(v, dim = -1)
        hs = F.softmax(h, dim = -2)
        # Axis-wise normalization
#         vs = self.normalize(vs, axis = -1)
#         hs = self.normalize(hs, axis = -2)
        # Axis-wise scaling
        vs = scale(vs, t_min = 3e-1)
        hs = scale(hs, t_min = 3e-1)
        # Applying moving average
        vs = self.ma_conv(vs)
        hs = self.ma_conv(hs.transpose(-1,-2)).transpose(-1,-2)
        # Axis-wise scaling
        vs = scale(vs, t_min = 3e-1)
        hs = scale(hs, t_min = 3e-1)
        # Tensor Multiplication
        x_a = torch.matmul(hs, vs)
        self.xam = x_a
        # Applying Hadamard product
        return torch.multiply(x, x_a)
        
    def forward(self, x):
        return self.attention_forward(x)
    
class RNN_ConvAttentionWrapper(nn.Module):
    '''
    Converts inputs to B,T,C,W,H for processing and returns original shape
    
    Uses V0 ConvAttentionWrapper
    '''
    def __init__(self, model, kernel_sizes = (3,5), projection = 64, out_channels = None, lambda_ = 1e-6, max_norm_clip = 1):
        super(RNN_ConvAttentionWrapper, self).__init__()
        self.model = model
        self.projection = projection
        self.kernel_sizes = kernel_sizes
        self.lambda_ = lambda_
        self.max_norm_clip = max_norm_clip
        self.out_channels = self.model.channels
        if out_channels != None: self.out_channels = out_channels
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.model.channels, self.projection, self.kernel_sizes[0], 1, padding = self.kernel_sizes[0]//2),
            nn.Conv2d(self.projection, self.out_channels, self.kernel_sizes[1], 1, padding = self.kernel_sizes[1]//2),
        )
        self.__name__ = self.model.__name__ + "_RNN_CONV_ATTENTION_P%s_L%s_C%s"%(self.projection, self.lambda_, self.out_channels)
        self.rnn_attention_conv = TimeDistributed(self.attention_conv)
        self.act_block = nn.Sequential(
            nn.BatchNorm2d(self.model.channels),
            nn.Sigmoid()
        )
        self.rnn_act_block = TimeDistributed(self.act_block)
    
    def attention_forward(self, x):
        # x -> bs,ts,c,w,h
        x_a = self.rnn_attention_conv(x)
        # x_a -> bs,ts,c,w,h
        return self.rnn_act_block(torch.multiply(x, x_a)) # output -> bs,tc,c,w,h
    
    def attention_loss(self, w):
        return self.lambda_ * torch.sqrt(torch.sum(w**2))
        
    def forward(self, x):
        # original x -> bs,c,ts,w,h
        x_a = self.attention_forward(x.permute(0,2,1,3,4))
        x_a = x_a.permute(0,2,1,3,4) # bs,c,ts,w,h
        model_returns = list(self.model(x_a))
        return tuple(model_returns + [x_a])

ConvAttentionWrapper = ConvAttentionWapper

class ConvAttentionLayerWrapper(nn.Module):
    def __init__(
        self,
        module,
        in_channels:int,
        out_channels:int,
        compression:int = 16,
        kernel_sizes:tuple = (3,5),
        stride:int = 1,
        padding = 0,
        lambda_:float = 1e-4
    ):
        super(ConvAttentionLayerWrapper, self).__init__()
        self.__name__ = "ConvAttentionLayerWrapper"
        self.module = module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.compression = compression
        self.stride = stride
        self.padding = padding
        self.kernel_sizes = kernel_sizes
        self.lambda_ = lambda_
        
        out_padding = (self.kernel_sizes[1]-1)//2 
        if self.kernel_sizes[1] % 2 == 0: out_padding -= 1
        
        self.attention_conv = nn.Sequential(
#             nn.Conv2d(self.in_channels, self.out_channels, self.kernel_sizes[0], stride, padding = self.padding),
            nn.Conv2d(self.in_channels, self.out_channels//self.compression, self.kernel_sizes[0], stride, padding = self.padding),
            nn.Conv2d(self.out_channels//self.compression, self.out_channels, self.kernel_sizes[1], 1, padding = out_padding),
        )
        self.act_block = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )
        
    def attention_loss(self):
        return self.lambda_ * torch.stack([torch.norm(layer.weight.data) for layer in self.attention_conv]).sum()
    
    def forward(self, x):
        module_output = self.module(x)
        attention_output = self.attention_conv(x)
        return self.act_block(torch.multiply(module_output, attention_output))

class ConvTransposeAttentionLayerWrapper(nn.Module):
    def __init__(
        self,
        module,
        in_channels:int,
        out_channels:int,
        compression:int = 16,
        kernel_sizes:tuple = (3,5),
        stride:int = 1,
        padding = 0,
        lambda_:float = 1e-6
    ):
        super(ConvTransposeAttentionLayerWrapper, self).__init__()
        self.__name__ = "ConvTransposeAttentionLayerWrapper"
        self.module = module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.compression = compression
        self.stride = stride
        self.padding = padding
        self.kernel_sizes = kernel_sizes
        self.lambda_ = lambda_
        
        out_padding = (self.kernel_sizes[1]-1)//2 
        if self.kernel_sizes[-1] % 2 == 0: out_padding -= 1
            
        self.attention_conv = nn.Sequential(
#             nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_sizes[0], stride, padding = self.padding),
            nn.ConvTranspose2d(self.in_channels, self.out_channels//self.compression, self.kernel_sizes[0], stride, padding = self.padding),
            nn.ConvTranspose2d(self.out_channels//self.compression, self.out_channels, self.kernel_sizes[1], 1, padding = out_padding),
        )
        self.act_block = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )
        
    def attention_loss(self):
        return self.lambda_ * torch.stack([torch.norm(layer.weight.data) for layer in self.attention_conv]).sum()
    
    def forward(self, x):
        module_output = self.module(x)
        attention_output = self.attention_conv(x)
        return self.act_block(torch.multiply(module_output, attention_output))
