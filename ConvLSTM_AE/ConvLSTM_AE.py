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
        self.clstm1 = Conv2dLSTM_Cell(self.image_size, self.channels, self.filters_count[0], 3, 2, 0, useGPU=useGPU)
        self.clstm1_os = getConvOutputShape(self.image_size, 3,2)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1_os, self.filters_count[0], self.filters_count[1], 3, 2, 0, useGPU=useGPU)
        self.clstm2_os = getConvOutputShape(self.clstm1_os, 3,2)
        self.clstm3 = Conv2dLSTM_Cell(self.clstm2_os, self.filters_count[1], self.filters_count[2], 3, 2, 0, useGPU=useGPU)
        self.clstm3_os = getConvOutputShape(self.clstm2_os, 3,2)
        self.clstm4 = Conv2dLSTM_Cell(self.clstm3_os, self.filters_count[2], self.filters_count[3], 3, 2, 0, useGPU=useGPU)
        self.clstm4_os = getConvOutputShape(self.clstm3_os, 3,2)
        self.clstm5 = Conv2dLSTM_Cell(self.clstm4_os, self.filters_count[3], self.filters_count[4], 4, 3, 0, useGPU=useGPU)
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
        self.clstm1 = Conv2dLSTM_Cell(self.image_size, self.channels, self.filters_count[0], 3, 2, 0, useGPU=useGPU)
        self.clstm1_os = getConvOutputShape(self.image_size, 3,2)
        self.clstm2 = Conv2dLSTM_Cell(self.clstm1_os, self.filters_count[0], self.filters_count[1], 3, 2, 0, useGPU=useGPU)
        self.clstm2_os = getConvOutputShape(self.clstm1_os, 3,2)
        self.clstm3 = Conv2dLSTM_Cell(self.clstm2_os, self.filters_count[1], self.filters_count[2], 3, 2, 0, useGPU=useGPU)
        self.clstm3_os = getConvOutputShape(self.clstm2_os, 3,2)
        self.clstm4 = Conv2dLSTM_Cell(self.clstm3_os, self.filters_count[2], self.filters_count[3], 3, 2, 0, useGPU=useGPU)
        self.clstm4_os = getConvOutputShape(self.clstm3_os, 3,2)
        self.clstm5 = Conv2dLSTM_Cell(self.clstm4_os, self.filters_count[3], self.filters_count[4], 4, 3, 0, useGPU=useGPU)
        self.clstm5_os = getConvOutputShape(self.clstm4_os, 4,3)
        
        # Decoder
        self.ctlstm6 = ConvTranspose2dLSTM_Cell(self.clstm5_os, self.filters_count[4], self.filters_count[3], 4, 3, 0, useGPU=useGPU)
        self.ctlstm6_os = getConvTransposeOutputShape(self.clstm5_os, 4,3)
        self.ctlstm7 = ConvTranspose2dLSTM_Cell(self.ctlstm6_os, self.filters_count[3], self.filters_count[2], 3, 2, 0, useGPU=useGPU)
        self.ctlstm7_os = getConvTransposeOutputShape(self.ctlstm6_os,3,2)
        self.ctlstm8 = ConvTranspose2dLSTM_Cell(self.ctlstm7_os, self.filters_count[2], self.filters_count[1], 3, 2, 0, useGPU=useGPU)
        self.ctlstm8_os = getConvTransposeOutputShape(self.ctlstm7_os,3,2)
        self.ctlstm9 = ConvTranspose2dLSTM_Cell(self.ctlstm8_os, self.filters_count[1], self.filters_count[0], 3, 2, 0, useGPU=useGPU)
        self.ctlstm9_os = getConvTransposeOutputShape(self.ctlstm8_os,3,2)
        self.ctlstm10 =ConvTranspose2dLSTM_Cell(self.ctlstm9_os, self.filters_count[0], self.channels, 4, 2, 0, useGPU=useGPU)
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
        "CLSTM_CTD": CLSTM_AE_CTD,
        "CLSTM": CLSTM_AE
    }
}