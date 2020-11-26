from .ConvLSTMCell import *

def getConvOutputShape(input_size, kernel_size, stride = 1, padding = 0):
    return ((input_size - kernel_size + (2 * padding)) // stride) + 1

class ConvLSTM_AE(nn.Module):
    def __init__(self, channels = 3, image_size = 128, hidden_dim = 64):
        super(ConvLSTM_AE, self).__init__()
        self.channels = channels
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        if not isinstance(self.hidden_dim, list): self.hidden_dim = [hidden_dim]*5
        
        self.encoder_CNN = nn.Conv3d(in_channels = self.channels,
                                     out_channels = self.hidden_dim[0],
                                     kernel_size = (1,3,3),
                                     stride = (1,2,2))
        self.eo_im = getConvOutputShape(image_size, 3, 2)
        self.encoder_clc1 = ConvLSTMCell(input_dim = self.hidden_dim[0],
                                        hidden_dim = self.hidden_dim[1],
                                        kernel_size = (3,3),
                                        bias = True)
        self.encoder_clc2 = ConvLSTMCell(input_dim = self.hidden_dim[1],
                                        hidden_dim = self.hidden_dim[2],
                                        kernel_size = (3,3),
                                        bias = True)
        self.decoder_clc1 = ConvLSTMCell(input_dim = self.hidden_dim[2],
                                        hidden_dim = self.hidden_dim[3],
                                        kernel_size = (3,3),
                                        bias = True)
        self.decoder_clc2 = ConvLSTMCell(input_dim = self.hidden_dim[3],
                                        hidden_dim = self.hidden_dim[4],
                                        kernel_size = (3,3),
                                        bias = True)
        self.decoder_CNN = nn.ConvTranspose3d(in_channels=self.hidden_dim[4],
                                     out_channels=self.channels,
                                     kernel_size=(1, 3, 3),
                                     stride = (1,2,2),
                                     output_padding = (0,1,1))
        self.final_act = nn.Sigmoid()
        
    def autoencoder(self, x, time_steps, hidden_list):
        outputs = list()
        hs_1, cs_1, hs_2, cs_2, hs_3, cs_3, hs_4, cs_4 = hidden_list
        
        x = self.encoder_CNN(x)
        x = x.permute(0, 2, 1, 3, 4) # bs, ts, c, h, w
        # Encode
        for ts in range(time_steps):
            hs_1, cs_1 = self.encoder_clc1(
                input_tensor = x[:, ts, :, :],
                cur_state = [hs_1, cs_1]
            )
            hs_2, cs_2 = self.encoder_clc2(
                input_tensor = hs_1,
                cur_state = [hs_2, cs_2]
            )
        
        encoding = hs_2
        # Decode
        for ts in range(time_steps):
            hs_3, cs_3 = self.decoder_clc1(
                input_tensor = encoding,
                cur_state = [hs_3, cs_3]
            )
            hs_4, cs_4 = self.decoder_clc2(
                input_tensor = hs_3,
                cur_state = [hs_4, cs_4]
            )
            encoding = hs_4
            outputs.append(hs_4)
            
        outputs = torch.stack(outputs, dim = 1).permute(0, 2, 1, 3, 4) # bs, c, ts, h, w
        outputs = self.decoder_CNN(outputs)
        outputs = self.final_act(outputs)
        return outputs, hs_2
    
    def forward(self, x, hidden_state = None):
        bs, c, ts, h, w = x.size()
        
        # Hidden states initialization
        [hs_1, cs_1] = self.encoder_clc1.init_hidden(batch_size = bs, image_size = (self.eo_im,self.eo_im))
        [hs_2, cs_2] = self.encoder_clc2.init_hidden(batch_size = bs, image_size = (self.eo_im,self.eo_im))
        [hs_3, cs_3] = self.decoder_clc1.init_hidden(batch_size = bs, image_size = (self.eo_im,self.eo_im))
        [hs_4, cs_4] = self.decoder_clc2.init_hidden(batch_size = bs, image_size = (self.eo_im,self.eo_im))
        
        hidden_list = [hs_1, cs_1, hs_2, cs_2, hs_3, cs_3, hs_4, cs_4]
        # autoencoder forward
        return self.autoencoder(x, ts, hidden_list)
    
class ConvLSTM_C3D_C2D_AE(nn.Module):
    def __init__(self, channels = 3, image_size = 128, hidden_dim = 64):
        super(ConvLSTM_C3D_C2D_AE, self).__init__()
        self.channels = channels
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        if not isinstance(self.hidden_dim, list): self.hidden_dim = [hidden_dim]*8
        
        self.encoder_CNN1 = nn.Conv3d(in_channels = self.channels,
                                     out_channels = self.hidden_dim[0],
                                     kernel_size = (1,3,3),
                                     stride = (1,2,2))
        self.ecnn1_os = getConvOutputShape(image_size, 3, 2)
        
        self.encoder_CNN2 = nn.Conv3d(in_channels = self.hidden_dim[0],
                                      out_channels = self.hidden_dim[1],
                                      kernel_size = (1,3,3),
                                      stride = (1,2,2))
        self.ecnn2_os = getConvOutputShape(self.ecnn1_os, 3, 2)
        
        self.encoder_clc1 = ConvLSTMCell(input_dim = self.hidden_dim[1],
                                        hidden_dim = self.hidden_dim[2],
                                        kernel_size = (3,3),
                                        bias = True)
        self.encoder_clc2 = ConvLSTMCell(input_dim = self.hidden_dim[2],
                                        hidden_dim = self.hidden_dim[3],
                                        kernel_size = (3,3),
                                        bias = True)
        
        self.inter_c2d1 = nn.Sequential(
            nn.Conv2d(in_channels = self.hidden_dim[5],
                      out_channels = self.hidden_dim[5],
                      kernel_size = 3,
                      stride = 2),
            nn.Conv2d(in_channels = self.hidden_dim[5],
                      out_channels = self.hidden_dim[5],
                      kernel_size = 2,
                      stride = 2),
            nn.BatchNorm2d(self.hidden_dim[5]),
            nn.LeakyReLU()
        )
        
        
        self.inter_c2d2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = self.hidden_dim[5],
                               out_channels = self.hidden_dim[5],
                               kernel_size = 3,
                               stride = 2),
            nn.ConvTranspose2d(in_channels = self.hidden_dim[5],
                               out_channels = self.hidden_dim[5],
                               kernel_size = 3,
                               stride = 2),
            nn.BatchNorm2d(self.hidden_dim[5]),
            nn.LeakyReLU()
        )
        
        self.decoder_clc1 = ConvLSTMCell(input_dim = self.hidden_dim[3],
                                        hidden_dim = self.hidden_dim[4],
                                        kernel_size = (3,3),
                                        bias = True)
        
        self.decoder_clc2 = ConvLSTMCell(input_dim = self.hidden_dim[4],
                                        hidden_dim = self.hidden_dim[5],
                                        kernel_size = (3,3),
                                        bias = True)
        
        self.decoder_CNN1 = nn.ConvTranspose3d(in_channels=self.hidden_dim[5],
                                     out_channels=self.hidden_dim[6],
                                     kernel_size=(1, 3, 3),
                                     stride = (1,2,2),)
        self.decoder_CNN2 = nn.ConvTranspose3d(in_channels=self.hidden_dim[6],
                                     out_channels=self.channels,
                                     kernel_size=(1, 3, 3),
                                     stride = (1,2,2),
                                     output_padding = (0,1,1))
        
        self.final_act = nn.Sigmoid()
        
    def autoencoder(self, x, time_steps, hidden_list):
        outputs = list()
        hs_1, cs_1, hs_2, cs_2, hs_3, cs_3, hs_4, cs_4 = hidden_list
        
        x = self.encoder_CNN1(x)
        x = self.encoder_CNN2(x)
        x = x.permute(0, 2, 1, 3, 4) # bs, ts, c, h, w
        
        # Encode
        for ts in range(time_steps):
            hs_1, cs_1 = self.encoder_clc1(
                input_tensor = x[:, ts, :, :],
                cur_state = [hs_1, cs_1]
            )
            hs_2, cs_2 = self.encoder_clc2(
                input_tensor = hs_1,
                cur_state = [hs_2, cs_2]
            )
        
        io1 = self.inter_c2d1(hs_2)
        encoding = io1
        io2 = self.inter_c2d2(io1)
        
        decoder_inputs = io2
        # Decode
        for ts in range(time_steps):
            hs_3, cs_3 = self.decoder_clc1(
                input_tensor = decoder_inputs,
                cur_state = [hs_3, cs_3]
            )
            hs_4, cs_4 = self.decoder_clc2(
                input_tensor = hs_3,
                cur_state = [hs_4, cs_4]
            )
            decoder_inputs = hs_4
            outputs.append(hs_4)
            
        outputs = torch.stack(outputs, dim = 1).permute(0, 2, 1, 3, 4) # bs, c, ts, h, w
        outputs = self.decoder_CNN1(outputs)
        outputs = self.decoder_CNN2(outputs)
        outputs = self.final_act(outputs)
        return outputs, encoding
    
    def forward(self, x, hidden_state = None):
        bs, c, ts, h, w = x.size()
        
        # Hidden states initialization
        [hs_1, cs_1] = self.encoder_clc1.init_hidden(batch_size = bs, image_size = (self.ecnn2_os,self.ecnn2_os))
        [hs_2, cs_2] = self.encoder_clc2.init_hidden(batch_size = bs, image_size = (self.ecnn2_os,self.ecnn2_os))
        [hs_3, cs_3] = self.decoder_clc1.init_hidden(batch_size = bs, image_size = (self.ecnn2_os,self.ecnn2_os))
        [hs_4, cs_4] = self.decoder_clc2.init_hidden(batch_size = bs, image_size = (self.ecnn2_os,self.ecnn2_os))
        
        hidden_list = [hs_1, cs_1, hs_2, cs_2, hs_3, cs_3, hs_4, cs_4]
        # autoencoder forward
        return self.autoencoder(x, ts, hidden_list)
    
    
class AllInOne(nn.Module):
    '''
    Encoding - SVM
    Noose
    Denoising
    Optical Flow support
    # Depth separable
    # pointwise
    # change conv lstm
    # add conv transpose lstm too
    # use two heads - one optical flow (shallow) and 1 for rgb (deep)
    # 1 or 2 steps ahead
    '''
    def __init__(self, channels = 3, image_size = 128, filters_count = [32, 64, 64, 64, 64], useGPU = True):
        super(AllInOne, self).__init__()
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available: self.device = torch.device("cuda")
        self.channels = channels
        self.image_size = image_size
        self.filters_count = filters_count
              
        self.encoder_clc1 = Conv2DLSTMCell_v1(
            image_size = self.image_size,
            input_dim = self.channels,
            hidden_dim = self.filters_count[0],
            kernel_size = 3,
            stride = 2,
            padding = 0
        ).to(self.device)
        self.encoder_clc1_os = getConvOutputShape(self.image_size, 3, 2)
        
        self.encoder_c3dres1 = C3D_Res(self.filters_count[0]).to(self.device)
        
        self.encoder_clc2 = Conv2DLSTMCell_v1(
            image_size = self.encoder_clc1_os,
            input_dim = self.filters_count[0],
            hidden_dim = self.filters_count[1],
            kernel_size = 3,
            stride = 2,
            padding = 0
        ).to(self.device)
        self.encoder_clc2_os = getConvOutputShape(self.encoder_clc1_os, 3, 2)
        
        self.encoder_c3d2 = C3D_BN_A(self.filters_count[1], self.filters_count[2], 3, 2).to(self.device)
        self.encoder_c3d2_os = getConvOutputShape(self.encoder_clc2_os, 3, 2)
        
        self.encoder_c3d3 = C3D_BN_A(self.filters_count[2], self.filters_count[3], 3, 2).to(self.device)
        self.encoder_c3d3_os = getConvOutputShape(self.encoder_c3d2_os, 3, 2)
        
        self.encoder_c3d4 = C3D_BN_A(self.filters_count[3], self.filters_count[4], 3, 2).to(self.device)
        self.encoder_c3d4_os = getConvOutputShape(self.encoder_c3d3_os, 3, 2)
        
        self.decoder_ct3d1 = CT3D_BN_A(self.filters_count[4], self.filters_count[3], 3, 2).to(self.device)
        self.decoder_ct3d1_os = getConvTransposeOutputShape(self.encoder_c3d4_os, 3, 2)
        
        self.decoder_ct3d2 = CT3D_BN_A(self.filters_count[3], self.filters_count[2], 3, 2).to(self.device)
        self.decoder_ct3d2_os = getConvTransposeOutputShape(self.decoder_ct3d1_os, 3, 2)
        
        self.decoder_ct3d3 = CT3D_BN_A(self.filters_count[2], self.filters_count[1], 4, 2).to(self.device)
        self.decoder_ct3d3_os = getConvTransposeOutputShape(self.decoder_ct3d2_os, 4, 2)
        
        self.decoder_clc4 = ConvTranspose2DLSTMCell_v1(
            image_size = self.decoder_ct3d3_os,
            input_dim = self.filters_count[1],
            hidden_dim = self.filters_count[0],
            kernel_size = 3,
            stride = 2,
            padding = 1
        ).to(self.device)
        self.decoder_clc4_os = getConvTransposeOutputShape(self.decoder_ct3d3_os, 3, 2, padding = 1)
        
        self.decoder_ct3d_res5 = CT3D_Res(self.filters_count[0]).to(self.device)
        
        self.decoder_clc5 = ConvTranspose2DLSTMCell_v1(
            image_size = self.decoder_clc4_os,
            input_dim = self.filters_count[0],
            hidden_dim = self.channels,
            kernel_size = 3,
            stride = 2,
            padding = 0
        ).to(self.device)
        self.decoder_clc5_os = getConvTransposeOutputShape(self.decoder_clc4_os, 3, 2)
    
    def clstm_bctwh(self, x):
        return x.transpose(0,1).transpose(1,2)
    
    def unroll_clstm(self, clstm, x, i_states, time_steps):
        clstm_states = list()
        hs,cs = i_states
        for ts in range(time_steps):
            hs, cs = clstm(
                x[:,:,ts,:,:],
                [hs, cs]
            )
            clstm_states.append(torch.stack([hs,cs]))
        clstm_states = torch.stack(clstm_states)
        clstm_h, clstm_c = self.clstm_bctwh(clstm_states[:,0,:,:,:]), self.clstm_bctwh(clstm_states[:,1,:,:,:])
        return clstm_h, clstm_c
    
    def autoencoder(self, x, time_steps, hidden_list):
        outputs = list()
        hs_1, cs_1, hs_2, cs_2, hs_3, cs_3, hs_4, cs_4 = hidden_list
        
        # Encoding
        clstm1_h, clstm1_c = self.unroll_clstm(self.encoder_clc1, x, [hs_1, cs_1], time_steps)
        c3d1_out = self.encoder_c3dres1(clstm1_h)
        clstm2_h, clstm2_c = self.unroll_clstm(self.encoder_clc2, c3d1_out, [hs_2, cs_2], time_steps)
        c3d2_out = self.encoder_c3d2(clstm2_h)
        c3d3_out = self.encoder_c3d3(c3d2_out)
        c3d4_out = self.encoder_c3d4(c3d3_out)
        
        encodings = c3d4_out
        
        # Decoding
        ct3d1_out = self.decoder_ct3d1(encodings)
        ct3d2_out = self.decoder_ct3d2(ct3d1_out)
        ct3d3_out = self.decoder_ct3d3(ct3d2_out)
        ctlstm4_h, ctlstm4_c = self.unroll_clstm(self.decoder_clc4, ct3d3_out, [hs_3, cs_3], time_steps)
        ct3d_res5_out = self.decoder_ct3d_res5(ctlstm4_h)
        ctlstm6_h, ctlstm6_c = self.unroll_clstm(self.decoder_clc5, ct3d_res5_out, [hs_4, cs_4], time_steps)
        
        return ctlstm6_h, encodings
        
    def forward(self, x):
        bs, c, ts, h, w = x.size()
        
        # Hidden states initialization
        [hs_1, cs_1] = self.encoder_clc1.init_states(batch_size = bs)
        [hs_2, cs_2] = self.encoder_clc2.init_states(batch_size = bs)
        [hs_3, cs_3] = self.decoder_clc4.init_states(batch_size = bs)
        [hs_4, cs_4] = self.decoder_clc5.init_states(batch_size = bs)
        
        hidden_list = [hs_1, cs_1, hs_2, cs_2, hs_3, cs_3, hs_4, cs_4]
        # autoencoder forward
        return self.autoencoder(x, ts, hidden_list)
    
class CLSTM_AE_CTD(nn.Module):
    def __init__(
        self,
        image_size,
        channels = 3,
        filters_count = [64,64,128,128,32], 
        useGPU = True
    ):
        super(CLSTM_AE_CTD, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        # Encoder
        self.clstm1 = Conv2DLSTMCell_v1(self.image_size, self.channels, self.filters_count[0], 3, 2, 0, useGPU=useGPU)
        self.clstm1_os = getConvOutputShape(self.image_size, 3,2)
        self.clstm2 = Conv2DLSTMCell_v1(self.clstm1_os, self.filters_count[0], self.filters_count[1], 3, 2, 0, useGPU=useGPU)
        self.clstm2_os = getConvOutputShape(self.clstm1_os, 3,2)
        self.clstm3 = Conv2DLSTMCell_v1(self.clstm2_os, self.filters_count[1], self.filters_count[2], 3, 2, 0, useGPU=useGPU)
        self.clstm3_os = getConvOutputShape(self.clstm2_os, 3,2)
        self.clstm4 = Conv2DLSTMCell_v1(self.clstm3_os, self.filters_count[2], self.filters_count[3], 3, 2, 0, useGPU=useGPU)
        self.clstm4_os = getConvOutputShape(self.clstm3_os, 3,2)
        self.clstm5 = Conv2DLSTMCell_v1(self.clstm4_os, self.filters_count[3], self.filters_count[4], 3, 2, 0, useGPU=useGPU)
        self.clstm5_os = getConvOutputShape(self.clstm4_os, 3,2)
        
        self.decoder = nn.Sequential(
            CT3D_BN_A(self.filters_count[4], self.filters_count[3], (1,3,3), (1,1,1)),
            CT3D_BN_A(self.filters_count[3], self.filters_count[2], (1,3,3), (1,2,2)),
            CT3D_BN_A(self.filters_count[2], self.filters_count[1], (1,5,5), (1,2,2), output_padding=(0,1,1)),
            CT3D_BN_A(self.filters_count[1], self.filters_count[0], (1,5,5), (1,2,2)),
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
        filters_count = [32,64,64,128], 
        useGPU = True
    ):
        super(CLSTM_AE, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.filters_count = filters_count
        
        # Encoder
        self.clstm1 = Conv2DLSTMCell_v1(self.image_size, self.channels, self.filters_count[0], 3, 2, 0, useGPU=useGPU)
        self.clstm1_os = getConvOutputShape(self.image_size, 3,2,0)
        self.clstm2 = Conv2DLSTMCell_v1(self.clstm1_os, self.filters_count[0], self.filters_count[1], 5, 3, 0, useGPU=useGPU)
        self.clstm2_os = getConvOutputShape(self.clstm1_os, 5,3,0)
        self.clstm3 = Conv2DLSTMCell_v1(self.clstm2_os, self.filters_count[1], self.filters_count[2], 5, 3, 0, useGPU=useGPU)
        self.clstm3_os = getConvOutputShape(self.clstm2_os, 5,3,0)
        self.clstm4 = Conv2DLSTMCell_v1(self.clstm3_os, self.filters_count[2], self.filters_count[3], 5, 2, 0, useGPU=useGPU)
        self.clstm4_os = getConvOutputShape(self.clstm3_os, 5,2,0)
               
#         Decoder
        self.ctlstm1 = ConvTranspose2DLSTMCell_v1(self.clstm4_os, self.filters_count[3], self.filters_count[3], 4, 3, useGPU=useGPU)
        self.ctlstm1_os = getConvTransposeOutputShape(self.clstm4_os, 4, 3)
        self.ctlstm2 = ConvTranspose2DLSTMCell_v1(self.ctlstm1_os, self.filters_count[3], self.filters_count[2], 4, 3, useGPU=useGPU)
        self.ctlstm2_os = getConvTransposeOutputShape(self.ctlstm1_os, 4, 3)
        self.ctlstm3 = ConvTranspose2DLSTMCell_v1(self.ctlstm2_os, self.filters_count[2], self.filters_count[1], 5, 3, useGPU=useGPU)
        self.ctlstm3_os = getConvTransposeOutputShape(self.ctlstm2_os, 5, 3)
        self.ctlstm4 = ConvTranspose2DLSTMCell_v1(self.ctlstm3_os, self.filters_count[1], self.filters_count[0], 5, 3, useGPU=useGPU)
        self.ctlstm4_os = getConvTransposeOutputShape(self.ctlstm3_os, 5, 3)
        self.ctlstm5 = ConvTranspose2DLSTMCell_v1(self.ctlstm4_os, self.filters_count[0], self.channels, 4, 1, useGPU=useGPU)
        self.ctlstm5_os = getConvTransposeOutputShape(self.ctlstm4_os, 4, 1)
    
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        
        [hs_1, cs_1] = self.clstm1.init_states(batch_size = bs)
        [hs_2, cs_2] = self.clstm2.init_states(batch_size = bs)
        [hs_3, cs_3] = self.clstm3.init_states(batch_size = bs)
        [hs_e, cs_e] = self.clstm4.init_states(batch_size = bs)
        
        [hs_6, cs_6] = self.ctlstm1.init_states(batch_size = bs)
        [hs_7, cs_7] = self.ctlstm2.init_states(batch_size = bs)
        [hs_8, cs_8] = self.ctlstm3.init_states(batch_size = bs)
        [hs_9, cs_9] = self.ctlstm4.init_states(batch_size = bs)
        [hs_o, cs_o] = self.ctlstm5.init_states(batch_size = bs)
        
        outputs = list()
        encodings = list()
        for t in range(ts):
            # Encoding process
            hs_1, cs_1 = self.clstm1(x[:,:,t,:,:], (hs_1, cs_1))
            hs_2, cs_2 = self.clstm2(hs_1, (hs_2, cs_2))
            hs_3, cs_3 = self.clstm3(hs_2, (hs_3, cs_3))
            hs_e, cs_e = self.clstm4(hs_3, (hs_e, cs_e))
            
            # Decoding process
            hs_6, cs_6 = self.ctlstm1(hs_e, (hs_6, cs_6))
            hs_7, cs_7 = self.ctlstm2(hs_6, (hs_7, cs_7))
            hs_8, cs_8 = self.ctlstm3(hs_7, (hs_8, cs_8))
            hs_9, cs_9 = self.ctlstm4(hs_8, (hs_9, cs_9))
            hs_o, cs_o = self.ctlstm5(hs_9, (hs_o, cs_o))
            
            encodings += [hs_e]
            outputs += [hs_o]
            
        encodings = torch.stack(encodings).permute(1,2,0,3,4)
        reconstructions = torch.stack(outputs).permute(1,2,0,3,4)
        return reconstructions, encodings