import sys
sys.path.append("..")
from general.model_utils import *
from general.all_imports import *

class CLSTM_CTD_AE(nn.Module):
    def __init__(self,
                 image_size = 128,
                 channels = 3,
                 filters_count = [256,128,128,128,128,128],
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
        
        self.clstm_act_blocks = nn.ModuleList([
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
                nn.LeakyReLU()
            ),
        ])
        
        self.encoding_layers = nn.ModuleList([self.clstm1, self.clstm2, self.clstm3, self.clstm4])
        
        self.decoder = nn.Sequential(
            CT3D_BN_A(self.filters_count[3], self.filters_count[2], (1,3,3), (1,2,2)),
            CT3D_BN_A(self.filters_count[2], self.filters_count[1], (1,5,5), (1,2,2)),
            CT3D_BN_A(self.filters_count[1], self.filters_count[1], (1,5,5), (1,3,3)),
            CT3D_BN_A(self.filters_count[1], self.filters_count[1], (2,5,5), (1,3,3)),
            CT3D_BN_A(self.filters_count[1], self.filters_count[0], (2,6,6), (1,1,1)),
            C3D_BN_A(self.filters_count[0], self.channels, 3, 1),
#             CT3D_BN_A(self.filters_count[1], self.filters_count[0], (2,5,5), (1,2,2)),
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
                 filters_count = [256,128,128,128,128,128],
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
                
        self.ctlstm1 = ConvTranspose2dLSTM_Cell(self.clstm2.output_shape, self.filters_count[3], self.filters_count[2], 4, 2)
        self.ctlstm2 = ConvTranspose2dLSTM_Cell(self.ctlstm1.output_shape, self.filters_count[2], self.filters_count[1], 4, 2)
        
        self.act_blocks = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[2]),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[3]),
                nn.LeakyReLU()
            ),
            
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[2]),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.BatchNorm2d(self.filters_count[1]),
                nn.LeakyReLU()
            ),
        ])
        
        self.lstm_layers = nn.Sequential(self.clstm1, self.clstm2, self.ctlstm1, self.ctlstm2)
        
        self.ct2d_1 = CT2D_BN_A(self.filters_count[1], self.filters_count[0], 5, 2)
        self.ct2d_2 = CT2D_BN_A(self.filters_count[0], self.filters_count[0], 5, 2)
        self.ct2d_3 = CT2D_BN_A(self.filters_count[0], self.filters_count[0], 3, 2)
        self.c2d_3 = C2D_BN_A(self.filters_count[0], self.channels, 4, 1)
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
    
class CLSTM_AE(nn.Module):
    def __init__(
        self,
        image_size,
        channels = 3,
        filters_count = [64,64,64,64,64]
    ):
        super(CLSTM_AE, self).__init__()
        self.__name__ = "CLSTM_128"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        # Encoder
        self.clstm1 = Conv2dLSTM_Cell(self.image_size, self.channels, self.filters_count[0], 3, 2, 0)
        self.clstm1_os = getConvOutputShape(self.image_size, 3,2)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1_os, self.filters_count[0], self.filters_count[1], 3, 2, 0)
        self.clstm2_os = getConvOutputShape(self.clstm1_os, 3,2)
        self.clstm3 = Conv2dLSTM_Cell(self.clstm2_os, self.filters_count[1], self.filters_count[2], 3, 2, 0)
        self.clstm3_os = getConvOutputShape(self.clstm2_os, 3,2)
        self.clstm4 = Conv2dLSTM_Cell(self.clstm3_os, self.filters_count[2], self.filters_count[3], 3, 2, 0)
        self.clstm4_os = getConvOutputShape(self.clstm3_os, 3,2)
        self.clstm5 = Conv2dLSTM_Cell(self.clstm4_os, self.filters_count[3], self.filters_count[4], 4, 3, 0)
        self.clstm5_os = getConvOutputShape(self.clstm4_os, 4,3)
        
        # Decoder
        self.ctlstm6 = ConvTranspose2dLSTM_Cell(self.clstm5_os, self.filters_count[4], self.filters_count[3], 4, 3, 0)
        self.ctlstm6_os = getConvTransposeOutputShape(self.clstm5_os, 4,3)
        self.ctlstm7 = ConvTranspose2dLSTM_Cell(self.ctlstm6_os, self.filters_count[3], self.filters_count[2], 3, 2, 0)
        self.ctlstm7_os = getConvTransposeOutputShape(self.ctlstm6_os,3,2)
        self.ctlstm8 = ConvTranspose2dLSTM_Cell(self.ctlstm7_os, self.filters_count[2], self.filters_count[1], 3, 2, 0)
        self.ctlstm8_os = getConvTransposeOutputShape(self.ctlstm7_os,3,2)
        self.ctlstm9 = ConvTranspose2dLSTM_Cell(self.ctlstm8_os, self.filters_count[1], self.filters_count[0], 3, 2, 0)
        self.ctlstm9_os = getConvTransposeOutputShape(self.ctlstm8_os,3,2)
        self.ctlstm10 =ConvTranspose2dLSTM_Cell(self.ctlstm9_os, self.filters_count[0], self.channels, 4, 2, 0)
        self.ctlstm10_os = getConvTransposeOutputShape(self.ctlstm9_os,4,2)
    
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        
        [hs_1, cs_1] = self.clstm1.init_states(batch_size = bs)
        [hs_2, cs_2] = self.clstm2.init_states(batch_size = bs)
        [hs_3, cs_3] = self.clstm3.init_states(batch_size = bs)
        [hs_4, cs_4] = self.clstm4.init_states(batch_size = bs)
        [hs_5, cs_5] = self.clstm5.init_states(batch_size = bs)
        
        [hs_6, cs_6] = self.ctlstm6.init_states(batch_size = bs)
        [hs_7, cs_7] = self.ctlstm7.init_states(batch_size = bs)
        [hs_8, cs_8] = self.ctlstm8.init_states(batch_size = bs)
        [hs_9, cs_9] = self.ctlstm9.init_states(batch_size = bs)
        [hs_10, cs_10] = self.ctlstm10.init_states(batch_size = bs)
        
        encodings = list()
        reconstructions = list()
        for t in range(ts):
            # Encoding process
            hs_1, cs_1 = self.clstm1(x[:,:,t,:,:], (hs_1, cs_1))
            hs_2, cs_2 = self.clstm2(hs_1, (hs_2, cs_2))
            hs_3, cs_3 = self.clstm3(hs_2, (hs_3, cs_3))
            hs_4, cs_4 = self.clstm4(hs_3, (hs_4, cs_4))
            hs_5, cs_5 = self.clstm5(hs_4, (hs_5, cs_5))
            
            hs_6, cs_6 = self.ctlstm6(hs_5, (hs_6, cs_6))
            hs_7, cs_7 = self.ctlstm7(hs_6, (hs_7, cs_7))
            hs_8, cs_8 = self.ctlstm8(hs_7, (hs_8, cs_8))
            hs_9, cs_9 = self.ctlstm9(hs_8, (hs_9, cs_9))
            hs_10, cs_10 = self.ctlstm10(hs_9, (hs_10, cs_10))
            
            encodings.append(hs_5)
            reconstructions.append(hs_10)
            
        encodings = torch.stack(encodings).transpose(0,1).transpose(1,2)
        reconstructions = torch.stack(reconstructions).transpose(0,1).transpose(1,2)
        return reconstructions, encodings    


    
CONV2D_LSTM_DICT = {
    128: {
        "CLSTM_CTD": CLSTM_CTD_AE,
        "CLSTM": CLSTM_AE,
        "CLSTM_C3D": CLSTM_C3D_AE,
        "CLSTM_C2D": CLSTM_C2D_AE
    }
}

'''
# Mixed ideas:

    3. Try expanding and contracting time steps
    
    4. Use ACB and Res Connections
    
'''