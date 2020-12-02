import torch
from torch import nn
import sys
sys.path.append("..")
from general.model_utils import *

class CLSTM_AE_CTD(nn.Module):
    def __init__(
        self,
        image_size,
        channels = 3,
        filters_count = [64,64,64,64,64], 
        useGPU = True
    ):
        super(CLSTM_AE_CTD, self).__init__()
        self.__name__ = "CLSTM_CTD_128"
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
        
        self.decoder = nn.Sequential(
            CT3D_BN_A(self.filters_count[4], self.filters_count[3], (1,4,4), (1,3,3)),
            CT3D_BN_A(self.filters_count[3], self.filters_count[2], (1,3,3), (1,2,2)),
            CT3D_BN_A(self.filters_count[2], self.filters_count[1], (1,3,3), (1,2,2)),
            CT3D_BN_A(self.filters_count[1], self.filters_count[0], (1,3,3), (1,2,2)),
            CT3D_BN_A(self.filters_count[0], self.channels, (1,4,4), (1,2,2)),
        )
    
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        
        [hs_1, cs_1] = self.clstm1.init_states(batch_size = bs)
        [hs_2, cs_2] = self.clstm2.init_states(batch_size = bs)
        [hs_3, cs_3] = self.clstm3.init_states(batch_size = bs)
        [hs_4, cs_4] = self.clstm4.init_states(batch_size = bs)
        [hs_5, cs_5] = self.clstm5.init_states(batch_size = bs)
        
        outputs = list()
        for t in range(ts):
            # Encoding process
            hs_1, cs_1 = self.clstm1(x[:,:,t,:,:], (hs_1, cs_1))
            hs_2, cs_2 = self.clstm2(hs_1, (hs_2, cs_2))
            hs_3, cs_3 = self.clstm3(hs_2, (hs_3, cs_3))
            hs_4, cs_4 = self.clstm4(hs_3, (hs_4, cs_4))
            hs_5, cs_5 = self.clstm5(hs_4, (hs_5, cs_5))
            outputs.append(hs_5)
        encodings = torch.stack(outputs).transpose(0,1).transpose(1,2)
        reconstructions = self.decoder(encodings)
        return reconstructions, encodings

class CLSTM_AE(nn.Module):
    def __init__(
        self,
        image_size,
        channels = 3,
        filters_count = [64,64,64,64,64], 
        useGPU = True
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

class CLSTM_C3D_AE(nn.Module):
    def __init__(
        self,
        image_size,
        channels = 3,
        filters_count = [64,64,64,64,64], 
        useGPU = True
    ):
        super(CLSTM_C3D_AE, self).__init__()
        self.__name__ = "CLSTM_C3D_128"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        # Encoder
        self.conv1 = C3D_BN_A(self.channels, self.filters_count[0], (1,3,3), (1,2,2))
        self.conv1_os = getConvOutputShape(self.image_size, 3,2)
        self.conv2 = C3D_BN_A(self.filters_count[0], self.filters_count[1], (1,3,3), (1,2,2))
        self.conv2_os = getConvOutputShape(self.conv1_os, 3,2)
        self.clstm1 = Conv2dLSTM_Cell(self.conv2_os, self.filters_count[1], self.filters_count[2], 4, 3, 0)
        self.clstm1_os = getConvOutputShape(self.conv2_os, 4,3)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1_os, self.filters_count[2], self.filters_count[3], 5, 3, 0)
        self.clstm2_os = getConvOutputShape(self.clstm1_os, 5,3)
                
        # Decoder
        self.ctlstm3 = ConvTranspose2dLSTM_Cell(self.clstm2_os, self.filters_count[3], self.filters_count[2], 4, 3, 0)
        self.ctlstm3_os = getConvTransposeOutputShape(self.clstm2_os, 4, 3)
        self.ctlstm4 = ConvTranspose2dLSTM_Cell(self.ctlstm3_os, self.filters_count[2], self.filters_count[1], 3, 2, 0)
        self.ctlstm4_os = getConvTransposeOutputShape(self.ctlstm3_os, 3, 2)
        self.convt_5 = CT3D_BN_A(self.filters_count[1], self.filters_count[0], (1,3,3), (1,2,2))
        self.convt_6 = CT3D_BN_A(self.filters_count[0], self.filters_count[0], (1,3,3), (1,2,2))
        self.convt_7 = CT3D_BN_A(self.filters_count[0], self.channels, (1,4,4), (1,2,2))
    
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        [hs_1, cs_1] = self.clstm1.init_states(batch_size = bs)
        [hs_2, cs_2] = self.clstm2.init_states(batch_size = bs)
        [hs_3, cs_3] = self.ctlstm3.init_states(batch_size = bs)
        [hs_4, cs_4] = self.ctlstm4.init_states(batch_size = bs)
        
        c1_o = self.conv1(x)
        c2_o = self.conv2(c1_o)
        
        encodings = list()
        clstm_out = list()
        for t in range(ts):
            # Encoding process
            hs_1, cs_1 = self.clstm1(c2_o[:,:,t,:,:], (hs_1, cs_1))
            hs_2, cs_2 = self.clstm2(hs_1, (hs_2, cs_2))
            hs_3, cs_3 = self.ctlstm3(hs_2, (hs_3, cs_3))
            hs_4, cs_4 = self.ctlstm4(hs_3, (hs_4, cs_4))
            encodings.append(hs_2)
            clstm_out.append(hs_4)
            
        encodings = torch.stack(encodings).transpose(0,1).transpose(1,2)
        clstm_out = torch.stack(clstm_out).transpose(0,1).transpose(1,2)
        ct_5_o = self.convt_5(clstm_out)
        ct_6_o = self.convt_6(ct_5_o)
        reconstructions =self.convt_7(ct_6_o)
        return reconstructions, encodings
    
class CLSTM_C2D_AE(nn.Module):
    '''
    conv reduce
    conv reduce
    conv lstm
    conv lstm
    deconv lstm 
    deconv lstm
    deconv expand
    deconv expand
    '''
    def __init__(
        self,
        image_size,
        channels = 3,
        filters_count = [64,64,64,64,64], 
        useGPU = True
    ):
        super(CLSTM_C2D_AE, self).__init__()
        self.__name__ = "CLSTM_MIXED_128"
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        # Encoder
        self.conv2d_1 = C2D_BN_A(self.channels, self.filters_count[0], 3, 2)
        self.conv2d_1_os = getConvOutputShape(self.image_size, 3, 2)
        self.conv2d_2 = C2D_BN_A(self.filters_count[0], self.filters_count[1], 3, 2)
        self.conv2d_2_os = getConvOutputShape(self.conv2d_1_os, 3, 2)
        
        self.clstm1 = Conv2dLSTM_Cell(self.conv2d_2_os, self.filters_count[1], self.filters_count[2], 4, 3, 0)
        self.clstm1_os = getConvOutputShape(self.conv2d_2_os, 4,3)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1_os, self.filters_count[2], self.filters_count[3], 5, 3, 0)
        self.clstm2_os = getConvOutputShape(self.clstm1_os, 5,3)
                
        # Decoder
        self.ctlstm3 = ConvTranspose2dLSTM_Cell(self.clstm2_os, self.filters_count[3], self.filters_count[2], 4, 3, 0)
        self.ctlstm3_os = getConvTransposeOutputShape(self.clstm2_os, 4, 3)
        self.ctlstm4 = ConvTranspose2dLSTM_Cell(self.ctlstm3_os, self.filters_count[2], self.filters_count[1], 3, 2, 0)
        self.ctlstm4_os = getConvTransposeOutputShape(self.ctlstm3_os, 3, 2)
        
        self.deconv = nn.Sequential(
            CT2D_BN_A(self.filters_count[1], self.filters_count[0], 3 ,2),
            CT2D_BN_A(self.filters_count[0], self.filters_count[0], 3 ,2),
            CT2D_BN_A(self.filters_count[0], self.channels, 4 ,2)
        )

    
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        
        [hs_1, cs_1] = self.clstm1.init_states(batch_size = bs)
        [hs_2, cs_2] = self.clstm2.init_states(batch_size = bs)
        [hs_3, cs_3] = self.ctlstm3.init_states(batch_size = bs)
        [hs_4, cs_4] = self.ctlstm4.init_states(batch_size = bs)
        
        inp = x.transpose(1,2).reshape(-1,c,w,h) # bsxts, c, w, h
        c1_o = self.conv2d_1(inp)
        c2_o = self.conv2d_2(c1_o)
        c2_o = c2_o.reshape(bs,ts,self.filters_count[1], self.conv2d_2_os, self.conv2d_2_os).transpose(1,2) # bs,c,ts,w,h
        
        encodings = list()
        clstm_out = list()
        for t in range(ts):
            # Encoding process
            hs_1, cs_1 = self.clstm1(c2_o[:,:,t,:,:], (hs_1, cs_1))
            hs_2, cs_2 = self.clstm2(hs_1, (hs_2, cs_2))
            hs_3, cs_3 = self.ctlstm3(hs_2, (hs_3, cs_3))
            hs_4, cs_4 = self.ctlstm4(hs_3, (hs_4, cs_4))
            encodings.append(hs_2)
            clstm_out.append(hs_4)
            
        encodings = torch.stack(encodings).transpose(0,1).transpose(1,2)
        clstm_out = torch.stack(clstm_out).transpose(0,1).transpose(1,2)
        
        clstm_out = clstm_out.transpose(1,2).reshape(-1,self.filters_count[1],self.ctlstm4_os,self.ctlstm4_os)
        reconstructions = self.deconv(clstm_out)
        reconstructions = reconstructions.reshape(bs,ts,c,w,h).transpose(1,2)
        return reconstructions, encodings
    
CONV2D_LSTM_DICT = {
    128: {
        "CLSTM_CTD": CLSTM_AE_CTD,
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