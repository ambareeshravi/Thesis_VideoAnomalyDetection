import sys
sys.path.append("..")
from general.model_utils import *
from general.all_imports import *

class CLSTM_CTD_AE(nn.Module):
    def __init__(self,
                 image_size = 128,
                 channels = 3,
                 filters_count = [64,64,64,128],
                ):
        super(CLSTM_CTD_AE, self).__init__()
        self.__name__ = "CLSTM_CTD_128"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        self.clstm1 = Conv2dLSTM_Cell(self.image_size, self.channels, self.filters_count[0], 3, 2)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1.output_shape, self.filters_count[0], self.filters_count[1], 3, 2)
        self.clstm3 = Conv2dLSTM_Cell(self.clstm2.output_shape, self.filters_count[1], self.filters_count[2], 5, 3)
        self.clstm4 = Conv2dLSTM_Cell(self.clstm3.output_shape, self.filters_count[2], self.filters_count[3], 3, 2)
        
        self.clstm_act_blocks = nn.Sequential(
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[0]),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[1]),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[2]),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[3]),
                nn.Tanh()
            ),
        )
        
        self.encoding_layers = nn.Sequential(self.clstm1, self.clstm2, self.clstm3, self.clstm4)
        
        self.decoder = nn.Sequential(
            CT3D_BN_A(self.filters_count[3], self.filters_count[2], (1,2,2), (1,1,1)),
            CT3D_BN_A(self.filters_count[2], self.filters_count[1], (1,5,5), (1,2,2)),
            CT3D_BN_A(self.filters_count[1], self.filters_count[1], (1,5,5), (1,3,3)),
            CT3D_BN_A(self.filters_count[1], self.filters_count[1], (2,5,5), (1,3,3)),
            CT3D_BN_A(self.filters_count[1], self.filters_count[0], (2,6,6), (1,1,1)),
            C3D_BN_A(self.filters_count[0], self.channels, 3, 1, activation_type = "sigmoid"),
        )
        
    def forward(self, x):
        bs, c, t, w, h = x.shape
        
        hidden_list = [None]*len(self.encoding_layers)
        
        encodings = list()
        for ts in range(t):
            ts_input = x[:,:,ts,:,:]
            for idx, layer in enumerate(self.encoding_layers):
                h_l, c_l = layer(ts_input, hidden_list[idx])
                h_l = self.clstm_act_blocks[idx](h_l)
                hidden_list[idx] = [h_l, c_l]
                ts_input = h_l
            encodings += [h_l]
        encodings = torch.stack(encodings).permute(1,2,0,3,4)
        
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings

class CLSTM_C2D_AE(nn.Module):
    def __init__(self,
                 image_size = 128,
                 channels = 3,
                 filters_count = [64,64,64,128],
                ):
        super(CLSTM_C2D_AE, self).__init__()
        self.__name__ = "CLSTM_CTD_128"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        self.c2d_encoder = nn.Sequential(
            TimeDistributed(C2D_BN_A(self.channels, self.filters_count[0], 3, 2)),
            TimeDistributed(C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2))
        )
        
        c2d_1_os = getConvOutputShape(self.image_size, 3, 2)
        c2d_2_os = getConvOutputShape(c2d_1_os, 3, 2)
        
        self.clstm1 = Conv2dLSTM_Cell(c2d_2_os, self.filters_count[1], self.filters_count[2], 5, 3)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1.output_shape, self.filters_count[2], self.filters_count[3], 3, 2)
        # Decoding part       
        self.ctlstm1 = ConvTranspose2dLSTM_Cell(self.clstm2.output_shape, self.filters_count[3], self.filters_count[2], 4, 1)
        self.ctlstm2 = ConvTranspose2dLSTM_Cell(self.ctlstm1.output_shape, self.filters_count[2], self.filters_count[1], 3, 2)
        
        self.act_blocks = nn.Sequential(
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[2]),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[3]),
                nn.Tanh()
            ),
            
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[2]),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[1]),
                nn.LeakyReLU()
            ),
        )
        
        self.lstm_layers = nn.Sequential(self.clstm1, self.clstm2, self.ctlstm1, self.ctlstm2)
        
        self.ct2d_decoder = nn.Sequential(
            TimeDistributed(CT2D_BN_A(self.filters_count[1], self.filters_count[0], 3, 2)),
            TimeDistributed(CT2D_BN_A(self.filters_count[0], self.filters_count[0], 4, 2)),
            TimeDistributed(CT2D_BN_A(self.filters_count[0], self.filters_count[0], 4, 2)),
            TimeDistributed(C2D_BN_A(self.filters_count[0], self.channels, 3, 1, activation_type = "sigmoid"))
        )
        
    def forward(self, x):
        bs,c,t,w,h = x.shape
        o_c2d = self.c2d_encoder(x.transpose(1,2)) # bs,t,c,w,h
        o_c2d = o_c2d.transpose(1,2)
        
        hidden_list = [None]*len(self.lstm_layers)
        
        encodings = list()
        outputs = list()
        for ts in range(t):
            ts_input = o_c2d[:,:,ts,:,:]
            for idx, layer in enumerate(self.lstm_layers):
                h_l, c_l = layer(ts_input, hidden_list[idx])
                h_l = self.act_blocks[idx](h_l)
                hidden_list[idx] = [h_l, c_l]
                ts_input = h_l
                if idx == ((len(self.lstm_layers)//2)-1): encodings += [h_l]
            outputs += [h_l]
        encodings = torch.stack(encodings).permute(1,2,0,3,4)
        lstm_outputs = torch.stack(outputs).permute(1,2,0,3,4)
        o_ct2d = self.ct2d_decoder(lstm_outputs.transpose(1,2))
        reconstructions = o_ct2d.transpose(1,2)
        return reconstructions, encodings
    
class CLSTM_C3D_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filters_count = [64,64,64,64,128]
    ):
        super(CLSTM_C3D_AE, self).__init__()
        self.__name__ = "CLSTM_C3D_128"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        self.clstm1 = Conv2dLSTM_Cell(self.image_size, self.channels, self.filters_count[0], 3, 2)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1.output_shape, self.filters_count[0], self.filters_count[1], 3, 2)
        
        self.clstm_layers = nn.Sequential(self.clstm1, self.clstm2)
        
        self.c3d_1 = C3D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2)
        self.c3d_2 = C3D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2)
        self.c3d_3 = C3D_BN_A(self.filters_count[3], self.filters_count[4], (2,4,4), (1,1,1), activation_type = "tanh")
        
        self.c3d_encoder = nn.Sequential(self.c3d_1, self.c3d_2, self.c3d_3)
        
        self.ct3d_1 = CT3D_BN_A(self.filters_count[4], self.filters_count[3], (3,4,4), (1,1,1))
        self.ct3d_2 = CT3D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2)
        self.ct3d_3 = CT3D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2)
        
        self.ct3d_decoder = nn.Sequential(self.ct3d_1, self.ct3d_2, self.ct3d_3)
        
        self.ctlstm1 = ConvTranspose2dLSTM_Cell(31, self.filters_count[1], self.filters_count[0], 4, 2)
        self.ctlstm2 = ConvTranspose2dLSTM_Cell(self.ctlstm1.output_shape, self.filters_count[0], self.filters_count[0], 4, 2)
        
        self.ctlstm_layers = nn.Sequential(self.ctlstm1, self.ctlstm2)
        
        self.c3d_4 = C3D_BN_A(self.filters_count[0], self.channels, (1,3,3), 1, activation_type = "sigmoid")
        
        self.act_blocks = nn.Sequential(
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[0]),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[1]),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[0]),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[0]),
                nn.LeakyReLU()
            )
        )
    
    def forward(self, x):
        bs, c, t, w, h = x.shape
        
        clstm_hidden_states = [None] * len(self.clstm_layers)
        clstm_outputs = list()
        for ts in range(t):
            ts_input = x[:,:,ts,:,:]
            for idx, layer in enumerate(self.clstm_layers):
                h_l, c_l = layer(ts_input, clstm_hidden_states[idx])
                h_l = self.act_blocks[idx](h_l)
                clstm_hidden_states[idx] = [h_l, c_l]
                ts_input = h_l
            clstm_outputs += [h_l]
        clstm_outputs = torch.stack(clstm_outputs).permute(1,2,0,3,4)
        
        c3d_outputs = self.c3d_encoder(clstm_outputs)
        encodings = c3d_outputs
        ct3d_outputs = self.ct3d_decoder(c3d_outputs)
        
        ctlstm_hidden_states = [None] * len(self.ctlstm_layers)
        ctlstm_outputs = list()
        for ts in range(t):
            ts_input = ct3d_outputs[:,:,ts,:,:]
            for idx, layer in enumerate(self.ctlstm_layers):
                h_l, c_l = layer(ts_input, ctlstm_hidden_states[idx])
                h_l = self.act_blocks[idx + 2](h_l)
                ctlstm_hidden_states[idx] = [h_l, c_l]
                ts_input = h_l
            ctlstm_outputs += [h_l]
        ctlstm_outputs = torch.stack(ctlstm_outputs).permute(1,2,0,3,4)
        reconstructions = self.c3d_4(ctlstm_outputs)
        return reconstructions, encodings
    
class CLSTM_FULL_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filters_count = [64,64,64,64,128]
    ):
        super(CLSTM_FULL_AE, self).__init__()
        self.__name__ = "CLSTM_FULL_128"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        # Encoder
        self.clstm1 = Conv2dLSTM_Cell(self.image_size, self.channels, self.filters_count[0], 3, 2, 0)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1.output_shape, self.filters_count[0], self.filters_count[1], 3, 2, 0)
        self.clstm3 = Conv2dLSTM_Cell(self.clstm2.output_shape, self.filters_count[1], self.filters_count[2], 3, 2, 0)
        self.clstm4 = Conv2dLSTM_Cell(self.clstm3.output_shape, self.filters_count[2], self.filters_count[3], 3, 2, 0)
        self.clstm5 = Conv2dLSTM_Cell(self.clstm4.output_shape, self.filters_count[3], self.filters_count[4], 4, 1, 0)
        
        # Decoder
        self.ctlstm6 = ConvTranspose2dLSTM_Cell(self.clstm5.output_shape, self.filters_count[4], self.filters_count[3], 4, 1, 0)
        self.ctlstm7 = ConvTranspose2dLSTM_Cell(self.ctlstm6.output_shape, self.filters_count[3], self.filters_count[2], 3, 2, 0)
        self.ctlstm8 = ConvTranspose2dLSTM_Cell(self.ctlstm7.output_shape, self.filters_count[2], self.filters_count[1], 3, 2, 0)
        self.ctlstm9 = ConvTranspose2dLSTM_Cell(self.ctlstm8.output_shape, self.filters_count[1], self.filters_count[0], 4, 2, 0)
        self.ctlstm10 =ConvTranspose2dLSTM_Cell(self.ctlstm9.output_shape, self.filters_count[0], self.filters_count[0], 4, 2, 0)
        self.clstm11 =Conv2dLSTM_Cell(self.ctlstm10.output_shape, self.filters_count[0], self.channels, 3, 1, 0)
        
        self.lstm_layers = nn.Sequential(
            self.clstm1, self.clstm2, self.clstm3, self.clstm4, self.clstm5,
            self.ctlstm6, self.ctlstm7, self.ctlstm8, self.ctlstm9, self.ctlstm10, self.clstm11
        )
        
        self.act_blocks = list()
        for i in [0,1,2,3]:
            self.act_blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.filters_count[i]),
                    nn.LeakyReLU()
                )
            )
        self.act_blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.filters_count[4]),
                    nn.Tanh()
                )
        )
        for i in [3,2,1,0,0]:
            self.act_blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.filters_count[i]),
                    nn.LeakyReLU()
                )
        )
        self.act_blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.channels),
                    nn.Sigmoid()
                )
        )
        self.act_blocks = nn.Sequential(*self.act_blocks)
        
    def forward(self, x, future_steps = 0):
        b,c,t,w,h = x.shape
        
        hidden_list = [None]*len(self.lstm_layers)
        
        encodings = list()
        reconstructions = list()
        
        for ts in range(t + future_steps):
            if ts < t: ts_input = x[:,:,ts,:,:]
            else: ts_input = prev_output
            for idx, layer in enumerate(self.lstm_layers):
                h_l, c_l = layer(ts_input, hidden_list[idx])
                h_l = self.act_blocks[idx](h_l)
                hidden_list[idx] = [h_l, c_l]
                ts_input = h_l
                if idx == 4: encodings += [h_l]
            reconstructions += [h_l]
            prev_output = h_l
            
        encodings = torch.stack(encodings).permute(1,2,0,3,4)
        reconstructions = torch.stack(reconstructions).permute(1,2,0,3,4)
        return reconstructions, encodings

class CLSTM_Multi_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filters_count = [64,64,64,64,128]
    ):
        super(CLSTM_Multi_AE, self).__init__()
        self.__name__ = "CLSTM_MULTI_128"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        # Encoder
        self.clstm1 = Conv2dLSTM_Cell(self.image_size, self.channels, self.filters_count[0], 3, 2, 0)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1.output_shape, self.filters_count[0], self.filters_count[1], 3, 2, 0)
        self.clstm3 = Conv2dLSTM_Cell(self.clstm2.output_shape, self.filters_count[1], self.filters_count[2], 3, 2, 0)
        self.clstm4 = Conv2dLSTM_Cell(self.clstm3.output_shape, self.filters_count[2], self.filters_count[3], 3, 2, 0)
        self.clstm5 = Conv2dLSTM_Cell(self.clstm4.output_shape, self.filters_count[3], self.filters_count[4], 3, 1, 0)
        
        # Decoder
        self.ctlstm6 = ConvTranspose2dLSTM_Cell(self.clstm5.output_shape, self.filters_count[4], self.filters_count[3], 3, 1, 0)
        self.ctlstm7 = ConvTranspose2dLSTM_Cell(self.ctlstm6.output_shape, self.filters_count[3], self.filters_count[2], 3, 2, 0)
        self.ctlstm8 = ConvTranspose2dLSTM_Cell(self.ctlstm7.output_shape, self.filters_count[2], self.filters_count[1], 3, 2, 0)
        self.ctlstm9 = ConvTranspose2dLSTM_Cell(self.ctlstm8.output_shape, self.filters_count[1], self.filters_count[0], 3, 2, 0)
        self.ctlstm10 =ConvTranspose2dLSTM_Cell(self.ctlstm9.output_shape, self.filters_count[0], self.channels, 4, 2, 0)
        
        self.lstm_layers = nn.Sequential(
            self.clstm1, self.clstm2, self.clstm3, self.clstm4, self.clstm5,
            self.ctlstm6, self.ctlstm7, self.ctlstm8, self.ctlstm9, self.ctlstm10
        )
        
        self.act_blocks = list()
        for i in [0,1,2,3]:
            self.act_blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.filters_count[i]),
                    nn.LeakyReLU()
                )
            )
        self.act_blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.filters_count[4]),
                    nn.Tanh()
                )
        )
        
        for i in [3,2,1,0]:
            self.act_blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.filters_count[i]),
                    nn.LeakyReLU()
                )
            )
        self.act_blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.channels),
                    nn.Sigmoid()
                )
        )
        self.act_blocks = nn.Sequential(*self.act_blocks)
        
    def forward(self, x, future_steps = 0):
        b,c,t,w,h = x.shape
        
        hidden_list = [None]*len(self.lstm_layers)
        
        encodings = list()
        reconstructions = list()
        
        for ts in range(t + future_steps):
            if ts < t: ts_input = x[:,:,ts,:,:]
            else: ts_input = prev_output
            for idx, layer in enumerate(self.lstm_layers):
                h_l, c_l = layer(ts_input, hidden_list[idx])
                h_l = self.act_blocks[idx](h_l)
                hidden_list[idx] = [h_l, c_l]
                ts_input = h_l
                if idx == 4: encodings += [h_l]
            reconstructions += [h_l]
            prev_output = h_l
        encodings = torch.stack(encodings).permute(1,2,0,3,4)
        reconstructions = torch.stack(reconstructions).permute(1,2,0,3,4)
        return reconstructions, encodings
    
class HashemCLSTM(nn.Module):
    def __init__(self, image_size = 256, channels = 3, filters_count = [128,64,32]):
        super(HashemCLSTM, self).__init__()
        self.__name__ = 'HashemCLSTM_256'
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        self.encoder_convs = nn.Sequential(
            TimeDistributed(C2D_BN_A(self.channels, self.filters_count[0], 11, 4)),
            TimeDistributed(C2D_BN_A(self.filters_count[0], self.filters_count[1], 5, 2)),
        )
        ec1_os = getConvOutputShape(self.image_size, 11, 4)
        ec2_os = getConvOutputShape(ec1_os, 5, 2)
        
        self.clstm1 = Conv2dLSTM_Cell(ec2_os, self.filters_count[1], self.filters_count[1], 3, 1, 1)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1.output_shape, self.filters_count[1], self.filters_count[2], 3, 1, 1)
        self.clstm3 = Conv2dLSTM_Cell(self.clstm2.output_shape, self.filters_count[2], self.filters_count[1], 3, 1, 1)
        
        self.lstm_layers = nn.ModuleList([self.clstm1, self.clstm2, self.clstm3])
        self.decoder_convs = nn.Sequential(
            TimeDistributed(CT2D_BN_A(self.filters_count[1], self.filters_count[0], 7, 2)),
            TimeDistributed(CT2D_BN_A(self.filters_count[0], self.channels, 12, 4)),
            TimeDistributed(C2D_BN_A(self.channels, self.channels, 5, 1))
        )
        
    def forward(self, x, future_steps = 0):
        # x - bs,c,ts,w,h        
        eo = self.encoder_convs(x.transpose(2,1))
        
        # eo - bs,ts,c,w,h
        bs,ts,c,w,h = eo.shape
        
        hidden_states = [None]*len(self.lstm_layers)
        outputs = list()
        encodings = list()
        for t in range(ts + future_steps):
            layer_input = eo[:,t,:,:,:]
            for idx, layer in enumerate(self.lstm_layers):
                h,c = layer(layer_input, hidden_states[idx])
                hidden_states[idx] = [h,c]
                if idx == 1: encodings += [h]
                layer_input = h
            outputs += [h]
        encodings = torch.stack(encodings).permute(1,2,0,3,4) # ts,bs,c,w,h -> bs,c,ts,w,h
        outputs = torch.stack(outputs) # ts,bs,c,w,h
        lstm_outputs = outputs.transpose(0,1)
        
        reconstructions = self.decoder_convs(lstm_outputs) # bs,ts,c,w,h\
        reconstructions = reconstructions.permute(0,2,1,3,4) # bs,c,ts,w,h
        return reconstructions, encodings

class C2D_LSTM_EN(nn.Module):
    def __init__(self, channels = 3, filters_count = [64,64,64,96,128], isBidirectional = False):
        super(C2D_LSTM_EN, self).__init__()
        self.__name__ = 'C2D_LSTM_EN'
        self.channels = channels
        self.filters_count = filters_count
        self.isBidirectional = isBidirectional
        
        self.encoder = nn.Sequential(
            TimeDistributed(C2D_BN_A(self.channels, self.filters_count[0], 3, 2)),
            TimeDistributed(C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2)),
            TimeDistributed(C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2)),
            TimeDistributed(C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2)),
            TimeDistributed(C2D_BN_A(self.filters_count[3], self.filters_count[4], 3, 1, activation_type="tanh")),
        )
        
        self.embedding_dim = [128, 5, 5]
        self.lstm_input_size = np.product(self.embedding_dim)
        self.lstm_hidden_size = np.product(self.embedding_dim)
        if self.isBidirectional:
            self.lstm_hidden_size = self.lstm_hidden_size // 2
        self.lstm = nn.LSTM(
            self.lstm_input_size,
            self.lstm_hidden_size,
            batch_first = True,
            bidirectional = isBidirectional
        )
        
        self.decoder = nn.Sequential(
            TimeDistributed(CT2D_BN_A(self.filters_count[4], self.filters_count[3], 3, 1)),
            TimeDistributed(CT2D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2)),
            TimeDistributed(CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2)),
            TimeDistributed(CT2D_BN_A(self.filters_count[1], self.filters_count[0], 3, 2)),
            TimeDistributed(CT2D_BN_A(self.filters_count[0], self.channels, 4, 2, activation_type="sigmoid")),
        )
        
    def get_states(self, bs):
        return torch.rand(1, bs, np.product(self.embedding_dim), device = self.encoder[0].module[0].weight.device), torch.rand(1, bs, np.product(self.embedding_dim), device = self.encoder[0].module[0].weight.device)
        
    def forward(self, x):
        xt = x.transpose(1,2)
        encoder_out = self.encoder(xt)
        lstm_in = encoder_out.flatten(start_dim=2, end_dim=-1)
        
        lstm_out, states = self.lstm(lstm_in, self.get_states(lstm_in.shape[0]))
        encodings = lstm_out.reshape(encoder_out.shape).transpose(1,2)
        reconstructions = self.decoder(encodings.transpose(1,2)).transpose(1,2)
        
        return reconstructions, encodings

class CLSTM_E1(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filters_count = [64,64,96,128]
    ):
        super(CLSTM_E1, self).__init__()
        self.__name__ = "CLSTM_E1"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        # Encoder
        self.encoder = nn.Sequential(
            TimeDistributed(C2D_BN_A(self.channels, self.filters_count[0], 3, 2)),
            TimeDistributed(C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2)),
            TimeDistributed(C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2)),
        )
        
        # CLSTM Encoder
        self.clstm1 = Conv2dLSTM_Cell(15, self.filters_count[2], self.filters_count[3], 5, 3, 0)
        self.clstm1_act_block = nn.Sequential(
            nn.BatchNorm2d(self.filters_count[3]),
            nn.Tanh()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            TimeDistributed(CT2D_BN_A(self.filters_count[3], self.filters_count[2], 6,3)),
            TimeDistributed(CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3,2)),
            TimeDistributed(CT2D_BN_A(self.filters_count[1], self.filters_count[0], 3,2)),
            TimeDistributed(CT2D_BN_A(self.filters_count[0], self.channels, 4,2, activation_type="sigmoid")),
        )
        
    def forward(self, x, future_steps = 0):
        b,c,t,w,h = x.shape
        
        c2d_encodings = self.encoder(x.transpose(1,2)) # bs, ts, nc, wn, hn
        c2d_encodings = c2d_encodings.transpose(1,2) # bs, nc, ts, wn, hn
        
        hidden_states = None
        encodings = list()
        for ts in range(c2d_encodings.shape[-3]):
            [h, c] = self.clstm1(c2d_encodings[:,:,ts,:,:], hidden_states)
            h = self.clstm1_act_block(h)
            hidden_states = [h, c]
            encodings += [h]
        
        encodings = torch.stack(encodings).permute(1,2,0,3,4)
        reconstructions = self.decoder(encodings.transpose(1,2)).transpose(1,2)
        return reconstructions, encodings
    
class CLSTM_E2(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filters_count = [64,64,96,128]
    ):
        super(CLSTM_E2, self).__init__()
        self.__name__ = "CLSTM_E2"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        # Encoder
        self.encoder = nn.Sequential(
            TimeDistributed(C2D_BN_A(self.channels, self.filters_count[0], 3, 2)),
            TimeDistributed(C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2)),
        )
        
        # CLSTM Encoder
        self.clstm1 = Conv2dLSTM_Cell(31, self.filters_count[1], self.filters_count[2], 3, 2, 0)
        self.clstm1_act_block = nn.Sequential(
            nn.BatchNorm2d(self.filters_count[2]),
            nn.LeakyReLU()
        )
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1.output_shape, self.filters_count[2], self.filters_count[3], 5, 3, 0)
        self.clstm2_act_block = nn.Sequential(
            nn.BatchNorm2d(self.filters_count[3]),
            nn.Tanh()
        )
        
        self.clstm_layers = nn.ModuleList([self.clstm1, self.clstm2])
        self.clstm_act_blocks = nn.ModuleList([self.clstm1_act_block, self.clstm2_act_block])
        
        # Decoder
        self.decoder = nn.Sequential(
            TimeDistributed(CT2D_BN_A(self.filters_count[3], self.filters_count[2], 6,3)),
            TimeDistributed(CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3,2)),
            TimeDistributed(CT2D_BN_A(self.filters_count[1], self.filters_count[0], 3,2)),
            TimeDistributed(CT2D_BN_A(self.filters_count[0], self.channels, 4,2, activation_type="sigmoid")),
        )
    
    def forward(self, x, future_steps = 0):
        b,c,t,w,h = x.shape
        
        c2d_encodings = self.encoder(x.transpose(1,2)) # bs, ts, nc, wn, hn
        c2d_encodings = c2d_encodings.transpose(1,2) # bs, nc, ts, wn, hn
        
        hidden_states = [None] * len(self.clstm_layers)
        encodings = list()
        
        for ts in range(t):
            layer_inputs = c2d_encodings[:,:,ts,:,:]
            for idx, (layer, layer_act_block) in enumerate(zip(self.clstm_layers, self.clstm_act_blocks)):
                [h, c] = layer(layer_inputs, hidden_states[idx])
                h = layer_act_block(h)
                hidden_states[idx] = [h, c]
                layer_inputs = h
            encodings += [h]
        
        encodings = torch.stack(encodings).permute(1,2,0,3,4)
        reconstructions = self.decoder(encodings.transpose(1,2)).transpose(1,2)
        return reconstructions, encodings

class CRNN_MULTI_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filters_count = [64,64,64,64,128]
    ):
        super(CRNN_MULTI_AE, self).__init__()
        self.__name__ = "CRNN_MULTI_AE"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        # Encoder
        self.crnn1 = Conv2dRNN_Cell(self.image_size, self.channels, self.filters_count[0], 3, 2, 0)
        self.crnn2 = Conv2dRNN_Cell(self.crnn1.output_shape, self.filters_count[0], self.filters_count[1], 3, 2, 0)
        self.crnn3 = Conv2dRNN_Cell(self.crnn2.output_shape, self.filters_count[1], self.filters_count[2], 3, 2, 0)
        self.crnn4 = Conv2dRNN_Cell(self.crnn3.output_shape, self.filters_count[2], self.filters_count[3], 3, 2, 0)
        self.crnn5 = Conv2dRNN_Cell(self.crnn4.output_shape, self.filters_count[3], self.filters_count[4], 3, 1, 0)
        
        # Decoder
        self.ctrnn6 = ConvTranspose2dRNN_Cell(self.crnn5.output_shape, self.filters_count[4], self.filters_count[3], 3, 1, 0)
        self.ctrnn7 = ConvTranspose2dRNN_Cell(self.ctrnn6.output_shape, self.filters_count[3], self.filters_count[2], 3, 2, 0)
        self.ctrnn8 = ConvTranspose2dRNN_Cell(self.ctrnn7.output_shape, self.filters_count[2], self.filters_count[1], 3, 2, 0)
        self.ctrnn9 = ConvTranspose2dRNN_Cell(self.ctrnn8.output_shape, self.filters_count[1], self.filters_count[0], 3, 2, 0)
        self.ctrnn10 =ConvTranspose2dRNN_Cell(self.ctrnn9.output_shape, self.filters_count[0], self.channels, 4, 2, 0)
        
        self.rnn_layers = nn.Sequential(
            self.crnn1, self.crnn2, self.crnn3, self.crnn4, self.crnn5,
            self.ctrnn6, self.ctrnn7, self.ctrnn8, self.ctrnn9, self.ctrnn10
        )
        
        self.act_blocks = list()
        for i in [0,1,2,3]:
            self.act_blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.filters_count[i]),
                    nn.LeakyReLU()
                )
            )
        self.act_blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.filters_count[4]),
                    nn.Tanh()
                )
        )
        
        for i in [3,2,1,0]:
            self.act_blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.filters_count[i]),
                    nn.LeakyReLU()
                )
            )
        self.act_blocks.append(
                nn.Sequential(
                    nn.BatchNorm2d(self.channels),
                    nn.Sigmoid()
                )
        )
        self.act_blocks = nn.Sequential(*self.act_blocks)
        
    def forward(self, x, future_steps = 0):
        b,c,t,w,h = x.shape
        
        hidden_list = [None]*len(self.rnn_layers)
        
        encodings = list()
        reconstructions = list()
        
        for ts in range(t + future_steps):
            if ts < t: ts_input = x[:,:,ts,:,:]
            else: ts_input = prev_output
            for idx, layer in enumerate(self.rnn_layers):
                h_l, y_l = layer(ts_input, hidden_list[idx])
                y_l = self.act_blocks[idx](y_l)
                hidden_list[idx] = h_l
                ts_input = y_l
                if idx == 4: encodings += [y_l]
            reconstructions += [y_l]
            prev_output = y_l
        encodings = torch.stack(encodings).permute(1,2,0,3,4)
        reconstructions = torch.stack(reconstructions).permute(1,2,0,3,4)
        return reconstructions, encodings
    
class BiCRNN_MULTI_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filters_count = [64,64,64,64,128]
    ):
        super(BiCRNN_MULTI_AE, self).__init__()
        self.__name__ = "BiCRNN_MULTI_AE"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        # Encoder
        self.crnn1_f = Conv2dRNN_Cell(self.image_size, self.channels, self.filters_count[0], 3, 2, 0)
        self.crnn1_b = Conv2dRNN_Cell(self.image_size, self.channels, self.filters_count[0], 3, 2, 0)
        
        self.crnn2_f = Conv2dRNN_Cell(self.crnn1_f.output_shape, self.filters_count[0], self.filters_count[1], 3, 2, 0)
        self.crnn2_b = Conv2dRNN_Cell(self.crnn1_b.output_shape, self.filters_count[0], self.filters_count[1], 3, 2, 0)
        
        self.crnn3_f = Conv2dRNN_Cell(self.crnn2_f.output_shape, self.filters_count[1], self.filters_count[2], 3, 2, 0)
        self.crnn3_b = Conv2dRNN_Cell(self.crnn2_b.output_shape, self.filters_count[1], self.filters_count[2], 3, 2, 0)
        
        self.crnn4_f = Conv2dRNN_Cell(self.crnn3_f.output_shape, self.filters_count[2], self.filters_count[3], 3, 2, 0)
        self.crnn4_b = Conv2dRNN_Cell(self.crnn3_b.output_shape, self.filters_count[2], self.filters_count[3], 3, 2, 0)
        
        self.crnn5_f = Conv2dRNN_Cell(self.crnn4_f.output_shape, self.filters_count[3], self.filters_count[4], 3, 1, 0)
        self.crnn5_b = Conv2dRNN_Cell(self.crnn4_b.output_shape, self.filters_count[3], self.filters_count[4], 3, 1, 0)
        
        # Decoder
        self.ctrnn6_f = ConvTranspose2dRNN_Cell(self.crnn5_f.output_shape, self.filters_count[4], self.filters_count[3], 3, 1, 0)
        self.ctrnn6_b = ConvTranspose2dRNN_Cell(self.crnn5_b.output_shape, self.filters_count[4], self.filters_count[3], 3, 1, 0)
        
        self.ctrnn7_f = ConvTranspose2dRNN_Cell(self.ctrnn6_f.output_shape, self.filters_count[3], self.filters_count[2], 3, 2, 0)
        self.ctrnn7_b = ConvTranspose2dRNN_Cell(self.ctrnn6_b.output_shape, self.filters_count[3], self.filters_count[2], 3, 2, 0)
        
        self.ctrnn8_f = ConvTranspose2dRNN_Cell(self.ctrnn7_f.output_shape, self.filters_count[2], self.filters_count[1], 3, 2, 0)
        self.ctrnn8_b = ConvTranspose2dRNN_Cell(self.ctrnn7_b.output_shape, self.filters_count[2], self.filters_count[1], 3, 2, 0)
        
        self.ctrnn9_f = ConvTranspose2dRNN_Cell(self.ctrnn8_f.output_shape, self.filters_count[1], self.filters_count[0], 3, 2, 0)
        self.ctrnn9_b = ConvTranspose2dRNN_Cell(self.ctrnn8_b.output_shape, self.filters_count[1], self.filters_count[0], 3, 2, 0)
        
        self.ctrnn10_f = ConvTranspose2dRNN_Cell(self.ctrnn9_f.output_shape, self.filters_count[0], self.channels, 4, 2, 0)
        self.ctrnn10_b = ConvTranspose2dRNN_Cell(self.ctrnn9_b.output_shape, self.filters_count[0], self.channels, 4, 2, 0)
        
        self.rnn_layers = [
            (self.crnn1_f, self.crnn1_b), (self.crnn2_f, self.crnn2_b), (self.crnn3_f, self.crnn3_b), (self.crnn4_f, self.crnn4_b), (self.crnn5_f, self.crnn5_b),
            (self.ctrnn6_f, self.ctrnn6_b), (self.ctrnn7_f, self.ctrnn7_b), (self.ctrnn8_f, self.ctrnn8_b), (self.ctrnn9_f, self.ctrnn9_b), (self.ctrnn10_f, self.ctrnn10_b)
        ]
        
        self.act_blocks_f = list()
        for i in [0,1,2,3]:
            self.act_blocks_f.append(BN_A(self.filters_count[i], is3d = False))
        self.act_blocks_f.append(BN_A(self.filters_count[4], activation_type = "tanh", is3d = False))
        
        for i in [3,2,1,0]:
            self.act_blocks_f.append(BN_A(self.filters_count[i], is3d = False))
        self.act_blocks_f.append(BN_A(self.channels, activation_type = "sigmoid", is3d = False))
        self.act_blocks_f = nn.Sequential(*self.act_blocks_f)
        
        self.act_blocks_b = list()
        for i in [0,1,2,3]:
            self.act_blocks_b.append(BN_A(self.filters_count[i], is3d = False))
        self.act_blocks_b.append(BN_A(self.filters_count[4], activation_type = "tanh", is3d = False))
        
        for i in [3,2,1,0]:
            self.act_blocks_b.append(BN_A(self.filters_count[i], is3d = False))
        self.act_blocks_b.append(BN_A(self.channels, activation_type = "sigmoid", is3d = False))
        self.act_blocks_b = nn.Sequential(*self.act_blocks_b)
        
        self.final_act_block = TimeDistributed(BN_A(self.channels, activation_type="sigmoid", is3d=False))
        
    def forward(self, x, future_steps = 0):
        b,c,t,w,h = x.shape
        
        hidden_list = [[None]*2]*len(self.rnn_layers)
        
        encodings_f = list()
        encodings_b = list()
        reconstructions_f = list()
        reconstructions_b = list()
        
        for ts in range(t):
            ip_f = x[:,:,ts,:,:]
            ip_b = x[:,:,t-(ts+1),:,:]
            for idx, (forward_layer, backward_layer) in enumerate(self.rnn_layers):
                h_f, y_f = forward_layer(ip_f, hidden_list[idx][0])
                h_b, y_b = backward_layer(ip_b, hidden_list[idx][1])
                hidden_list[idx] = [h_f, h_b]
                o_f, o_b = self.act_blocks_f[idx](y_f), self.act_blocks_b[idx](y_b)
                ip_f, ip_b = o_f, o_b
                if idx == 4:
                    encodings_f += [o_f]
                    encodings_b += [o_b]
            reconstructions_f += [o_f]
            reconstructions_b += [o_b]
        
        encodings_f = torch.stack(encodings_f)
        encodings_b = torch.stack(encodings_b)
        encodings = (encodings_f + torch.fliplr(encodings_b))
        encodings = encodings.permute(1,2,0,3,4)
        
        reconstructions_f = torch.stack(reconstructions_f)
        reconstructions_b = torch.stack(reconstructions_b)
        reconstructions = self.final_act_block(reconstructions_f + torch.fliplr(reconstructions_b))
        reconstructions = reconstructions.permute(1,2,0,3,4)
        
        return reconstructions, encodings
    
CONV2D_LSTM_DICT = {
    128: {
        "CLSTM_CTD": CLSTM_CTD_AE,
        "CLSTM_C3D": CLSTM_C3D_AE,
        "CLSTM_C2D": CLSTM_C2D_AE,
        "CLSTM_FULL": CLSTM_FULL_AE,
        "CLSTM_Multi": CLSTM_Multi_AE,
        "C2D_LSTM_EN": C2D_LSTM_EN,
        "CLSTM_E1": CLSTM_E1,
        "CLSTM_E2": CLSTM_E2
    },
    256: {
        "hashem": HashemCLSTM
    }
}

CONV2D_RNN_DICT = {
    128: {
        "CRNN_MULTI": CRNN_MULTI_AE,
        "BiCRNN_MULTI_AE": BiCRNN_MULTI_AE
    }
}
'''
# Mixed ideas:

    3. Try expanding and contracting time steps
    
    4. Use ACB and Res Connections    
'''

class FUNNEL(nn.Module):
    def __init__(
        self,
        isRNN = True,
        image_size = 128,
        channels = 3,
        filters_count = [64,64,96,96,128,128]
    ):
        super(FUNNEL, self).__init__()
        self.__name__ = "FUNNEL"
        self.isRNN = isRNN
        if self.isRNN:
            self.__name__ += "_CRNN"
            self.input_layer = Conv2dRNN_Cell
            self.output_layer = ConvTranspose2dRNN_Cell
        else:
            self.__name__ += "_CLSTM"
            self.input_layer = Conv2dLSTM_Cell
            self.output_layer = ConvTranspose2dLSTM_Cell
            
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        self.i_layer = self.input_layer(self.image_size, self.channels, self.filters_count[0], kernel_size=3, stride=2, conv_bias=True, init_random=True)
        self.input_act_block = BN_A(self.filters_count[0], is3d=False)
        
        self.encoder_layers = nn.Sequential(
            C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2),
            C2D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2),
            C2D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2),
            C2D_BN_A(self.filters_count[3], self.filters_count[4], 4, 1, activation_type="tanh"),
        )
        
        current_shape = self.i_layer.output_shape
        for k,s in zip([3,3,3,3], [2,2,2,1]):
            current_shape = getConvOutputShape(current_shape, k, s)
            
        self.decoder_layers = nn.Sequential(
            CT2D_BN_A(self.filters_count[4], self.filters_count[3], 4, 1),
            CT2D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2),
            CT2D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2),
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 3, 2),
        )
        
        for k,s in zip([3,3,3,3], [1,2,2,2]):
            current_shape = getConvTransposeOutputShape(current_shape, k, s)
            
        self.o_layer = self.output_layer(current_shape, self.filters_count[0], self.channels, kernel_size=4, stride = 2, conv_bias=True, init_random=True)
        self.output_act_block = BN_A(self.channels, activation_type="sigmoid", is3d=False)
    
    def temporal_squeeze(self, x):
        # bs,ts,c,w,h = x.shape
        return torch.stack([ts_x for bs_x in x for ts_x in bs_x])
    
    def temporal_expand(self, x, bs, ts = 16):
        # bs*ts,c,w,h = x.shape
        return torch.stack([torch.stack([x[t_idx] for t_idx in range(ts)]) for b_idx in range(bs)])
        
    def forward(self, x):
        bs, c, ts, w, h = x.shape
        
        input_hidden, output_hidden = None, None
        
        l1_output = list()
        for t in range(ts):
            y_n, input_hidden = self.i_layer(x[:,:,t,:,:], input_hidden)
            y_n = self.input_act_block(y_n)
            l1_output.append(y_n)
            
        l1_output = torch.stack(l1_output).transpose(0,1) # bs,ts,c,w,h
        l1_output = self.temporal_squeeze(l1_output) # bs * ts, c, w, h
        encodings = self.encoder_layers(l1_output)
        decoded_output = self.decoder_layers(encodings)
        decoded_output = self.temporal_expand(decoded_output, bs = bs, ts = ts) # bs,ts,c,w,h
        
        final_output = list()
        for t in range(ts):
            y_n, output_hidden = self.o_layer(decoded_output[:,t,:,:,:], output_hidden)
            y_n = self.output_act_block(y_n)
            final_output.append(y_n)
        reconstructions = torch.stack(final_output) # ts, bs, c, w, h
        reconstructions = reconstructions.transpose(0,1).transpose(1,2)
        encodings = self.temporal_expand(encodings, bs = bs, ts = ts).transpose(1,2) # bs,ts,c,w,h -> bs,c,ts,w,h
        return reconstructions, encodings