import sys
sys.path.append("..")
from general.model_utils import *
from general.all_imports import *

class CLSTM_CTD_AE(nn.Module):
    def __init__(self,
                 image_size = 128,
                 channels = 3,
                 filters_count = [64,64,64,128,128,128],
                ):
        super(CLSTM_CTD_AE, self).__init__()
        self.__name__ = "CLSTM_CTD_128"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        self.clstm1 = Conv2dLSTM_Cell(self.image_size, self.channels, self.filters_count[0], 3, 2)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1.output_shape, self.filters_count[0], self.filters_count[1], 3, 2)
        self.clstm3 = Conv2dLSTM_Cell(self.clstm2.output_shape, self.filters_count[1], self.filters_count[2], 5, 3)
        self.clstm4 = Conv2dLSTM_Cell(self.clstm3.output_shape, self.filters_count[2], self.filters_count[3], 5, 3)
        
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
            CT3D_BN_A(self.filters_count[3], self.filters_count[2], (1,3,3), (1,2,2)),
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
                 filters_count = [64,64,64,128,128,128],
                ):
        super(CLSTM_C2D_AE, self).__init__()
        self.__name__ = "CLSTM_CTD_128"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        self.c2d_1 = C2D_BN_A(self.channels, self.filters_count[0], 3, 2)
        self.c2d_1.output_shape = getConvOutputShape(self.image_size, 3, 2)
        self.c2d_2 = C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2)
        self.c2d_2.output_shape = getConvOutputShape(self.c2d_1.output_shape, 3, 2)
        
        self.c2d_encoder = nn.Sequential(self.c2d_1, self.c2d_2)
        
        self.clstm1 = Conv2dLSTM_Cell(self.c2d_encoder[-1].output_shape, self.filters_count[1], self.filters_count[2], 5, 3)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1.output_shape, self.filters_count[2], self.filters_count[3], 5, 3)
        # Decoding part       
        self.ctlstm1 = ConvTranspose2dLSTM_Cell(self.clstm2.output_shape, self.filters_count[3], self.filters_count[2], 4, 2)
        self.ctlstm2 = ConvTranspose2dLSTM_Cell(self.ctlstm1.output_shape, self.filters_count[2], self.filters_count[1], 4, 2)
        
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
        
        self.ct2d_1 = CT2D_BN_A(self.filters_count[1], self.filters_count[0], 5, 2)
        self.ct2d_2 = CT2D_BN_A(self.filters_count[0], self.filters_count[0], 5, 2)
        self.ct2d_3 = CT2D_BN_A(self.filters_count[0], self.filters_count[0], 3, 2)
        self.c2d_3 = C2D_BN_A(self.filters_count[0], self.channels, 4, 1, activation_type = "sigmoid")
        self.ct2d_decoder = nn.Sequential(self.ct2d_1, self.ct2d_2, self.ct2d_3, self.c2d_3)
        
    def forward(self, x):
        bs, c, t, w, h = x.shape
        x_c2d = x.transpose(1,2).flatten(0,1)
        o_c2d = self.c2d_encoder(x_c2d)
        o_c2d = o_c2d.reshape(bs, -1, t, self.c2d_encoder[-1].output_shape, self.c2d_encoder[-1].output_shape)
        
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
        lstm_flat = lstm_outputs.transpose(1,2).flatten(0,1)
        reconstructions = self.ct2d_decoder(lstm_flat)
        reconstructions = reconstructions.reshape(bs,t,c,w,h).transpose(1,2)
        return reconstructions, encodings
    
class CLSTM_C3D_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filters_count = [64,64,64,128,128,128]
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
        self.c3d_3 = C3D_BN_A(self.filters_count[3], self.filters_count[4], (2,4,4), (1,3,3), activation_type = "tanh")
        
        self.c3d_encoder = nn.Sequential(self.c3d_1, self.c3d_2, self.c3d_3)
        
        self.ct3d_1 = CT3D_BN_A(self.filters_count[4], self.filters_count[3], (3,4,4), (1,3,3))
        self.ct3d_2 = CT3D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2)
        self.ct3d_3 = CT3D_BN_A(self.filters_count[2], self.filters_count[1], 3, 2)
        
        self.ct3d_decoder = nn.Sequential(self.ct3d_1, self.ct3d_2, self.ct3d_3)
        
        self.ctlstm1 = ConvTranspose2dLSTM_Cell(31, self.filters_count[1], self.filters_count[0], 3, 2)
        self.ctlstm2 = ConvTranspose2dLSTM_Cell(self.ctlstm1.output_shape, self.filters_count[0], self.filters_count[0], 6, 2)
        
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
        filters_count = [64,64,64,128,128,128]
    ):
        super(CLSTM_FULL_AE, self).__init__()
        self.__name__ = "CLSTM_FULL_128_AE"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        # Encoder
        self.clstm1 = Conv2dLSTM_Cell(self.image_size, self.channels, self.filters_count[0], 3, 2, 0)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1.output_shape, self.filters_count[0], self.filters_count[1], 3, 2, 0)
        self.clstm3 = Conv2dLSTM_Cell(self.clstm2.output_shape, self.filters_count[1], self.filters_count[2], 3, 2, 0)
        self.clstm4 = Conv2dLSTM_Cell(self.clstm3.output_shape, self.filters_count[2], self.filters_count[3], 3, 2, 0)
        self.clstm5 = Conv2dLSTM_Cell(self.clstm4.output_shape, self.filters_count[3], self.filters_count[4], 4, 3, 0)
        
        # Decoder
        self.ctlstm6 = ConvTranspose2dLSTM_Cell(self.clstm5.output_shape, self.filters_count[4], self.filters_count[3], 4, 3, 0)
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
        
    def forward(self, x):
        b,c,t,w,h = x.shape
        
        hidden_list = [None]*len(self.lstm_layers)
        
        encodings = list()
        reconstructions = list()
        
        for ts in range(t):
            ts_input = x[:,:,ts,:,:]
            for idx, layer in enumerate(self.lstm_layers):
                h_l, c_l = layer(ts_input, hidden_list[idx])
                h_l = self.act_blocks[idx](h_l)
                hidden_list[idx] = [h_l, c_l]
                ts_input = h_l
                if idx == 4: encodings += [h_l]
            reconstructions += [h_l]
        encodings = torch.stack(encodings).permute(1,2,0,3,4)
        reconstructions = torch.stack(reconstructions).permute(1,2,0,3,4)
        return reconstructions, encodings    

CONV2D_LSTM_DICT = {
    128: {
        "CLSTM_CTD": CLSTM_CTD_AE,
        "CLSTM_FULL": CLSTM_FULL_AE,
        "CLSTM_C3D": CLSTM_C3D_AE,
        "CLSTM_C2D": CLSTM_C2D_AE
    }
}

'''
# Mixed ideas:

    3. Try expanding and contracting time steps
    
    4. Use ACB and Res Connections    
'''