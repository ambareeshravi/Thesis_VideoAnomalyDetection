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
        self.__name__ = "CLSTM_FULL_128"
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

'''
# Mixed ideas:

    3. Try expanding and contracting time steps
    
    4. Use ACB and Res Connections    
'''