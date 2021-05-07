from .recurrent_convs import *

# Recurrent Conv AutoEncoders
class CRNN_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filter_count = [64]*3 + [96]*2,
        filter_sizes =[3]*5,
        filter_strides = [2]*4 + [1],
        n_r_layers = 2,
        disableRecDeConv = True,
        useBias = False
    ):
        super(CRNN_AE, self).__init__()
        self.__name__ = "CRNN_AE_%d_%dx%d_L-%d_RL-%d_DisDeConv-%s-"%(image_size, most_common(filter_sizes), most_common(filter_sizes), len(filter_count), n_r_layers, "Y" if disableRecDeConv else "N")
        self.channels = channels
        self.image_size = image_size
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableRecDeConv = disableRecDeConv
        
        self.n_layers = len(self.filter_count)
        self.n_r_layers = n_r_layers
        self.n_normal = self.n_layers - self.n_r_layers
        
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_r_layers:
                insert = TimeDistributed(C2D_BN_A(in_channels, n, k, s))
            else:
                insert = ConvRNN_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias)
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
            if idx < self.n_r_layers and not self.disableRecDeConv:
                insert = ConvTransposeRNN_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
        
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)
                                
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_r_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h
        
        if self.n_r_layers != 0:
            states_list = [None] * 2 * self.n_r_layers
            current_input = preliminary_encodings
            rnn_outputs = list()
            rnn_layers = self.encoder_layers[self.n_normal:]
            if not self.disableRecDeConv: rnn_layers += self.decoder_layers[:self.n_r_layers]
            for idx, layer in enumerate(rnn_layers):
                layer_outputs = list()
                states = states_list[idx]
                for t in range(ts):
                    y, h = layer(current_input[:,t,...], states)
                    layer_outputs.append(y)
                    states = h
                layer_output = torch.stack(layer_outputs, dim = 1) # b,ts,c,w,h
                rnn_outputs.append(layer_output)
                current_input = layer_output
                states_list[idx] = states
            encodings = rnn_outputs[self.n_r_layers - 1].transpose(1,2)
        else:
            layer_output = preliminary_encodings
            encodings = layer_output
        
        decode_index = self.n_r_layers
        if self.disableRecDeConv: decode_index = 0
        reconstructions = nn.Sequential(*self.decoder_layers[decode_index:])(layer_output)
        reconstructions = reconstructions.transpose(1,2)
        return reconstructions, encodings
    
class CGRU_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filter_count = [64]*3 + [96]*2,
        filter_sizes =[3]*5,
        filter_strides = [2]*4 + [1],
        n_r_layers = 2,
        disableRecDeConv = True,
        useBias = False
    ):
        super(CGRU_AE, self).__init__()
        self.__name__ = "CGRU_AE_%d_%dx%d_L-%d_RL-%d_DisDeConv-%s-"%(image_size, most_common(filter_sizes), most_common(filter_sizes), len(filter_count), n_r_layers, "Y" if disableRecDeConv else "N")
        self.channels = channels
        self.image_size = image_size
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableRecDeConv = disableRecDeConv
        
        self.n_layers = len(self.filter_count)
        self.n_r_layers = n_r_layers
        self.n_normal = self.n_layers - self.n_r_layers
        
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_r_layers:
                insert = TimeDistributed(C2D_BN_A(in_channels, n, k, s))
            else:
                insert = ConvGRU_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias)
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
            if idx < self.n_r_layers and not self.disableRecDeConv:
                insert = ConvTransposeGRU_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
        
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)
                                
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_r_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h
        
        if self.n_r_layers != 0:
            states_list = [None] * 2 * self.n_r_layers
            current_input = preliminary_encodings
            gru_outputs = list()
            gru_layers = self.encoder_layers[self.n_normal:]
            if not self.disableRecDeConv: gru_layers += self.decoder_layers[:self.n_r_layers]
            for idx, layer in enumerate(gru_layers):
                layer_outputs = list()
                states = states_list[idx]
                for t in range(ts):
                    y, h = layer(current_input[:,t,...], states)
                    layer_outputs.append(y)
                    states = h
                layer_output = torch.stack(layer_outputs, dim = 1) # b,ts,c,w,h
                gru_outputs.append(layer_output)
                current_input = layer_output
                states_list[idx] = states
            encodings = gru_outputs[self.n_r_layers - 1].transpose(1,2)
        else:
            layer_output = preliminary_encodings
            encodings = layer_output
        
        decode_index = self.n_r_layers
        if self.disableRecDeConv: decode_index = 0
        reconstructions = nn.Sequential(*self.decoder_layers[decode_index:])(layer_output)
        reconstructions = reconstructions.transpose(1,2)
        return reconstructions, encodings
    
class CLSTM_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filter_count = [64]*3 + [96]*2,
        filter_sizes =[3]*5,
        filter_strides = [2]*4 + [1],
        n_r_layers = 2,
        disableRecDeConv = True,
        useBias = False
    ):
        super(CLSTM_AE, self).__init__()
        self.__name__ = "CLSTM_AE_%d_%dx%d_L-%d_RL-%d_DisDeConv-%s-"%(image_size, most_common(filter_sizes), most_common(filter_sizes), len(filter_count), n_r_layers, "Y" if disableRecDeConv else "N")
        self.channels = channels
        self.image_size = image_size
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableRecDeConv = disableRecDeConv
        
        self.n_layers = len(self.filter_count)
        self.n_r_layers = n_r_layers
        self.n_normal = self.n_layers - self.n_r_layers
        
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_r_layers:
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
            if idx < self.n_r_layers and not self.disableRecDeConv:
                insert = ConvTransposeLSTM_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
        
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)
                                
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_r_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h
        
        if self.n_r_layers != 0:
            states_list = [None] * 2 * self.n_r_layers
            current_input = preliminary_encodings
            lstm_outputs = list()
            lstm_layers = self.encoder_layers[self.n_normal:]
            if not self.disableRecDeConv: lstm_layers += self.decoder_layers[:self.n_r_layers]
            for idx, layer in enumerate(lstm_layers):
                layer_outputs = list()
                states = states_list[idx]
                for t in range(ts):
                    states = layer(current_input[:,t,...], states)
                    layer_outputs.append(states[0])
                layer_output = torch.stack(layer_outputs, dim = 1) # b,ts,c,w,h
                lstm_outputs.append(layer_output)
                current_input = layer_output
                states_list[idx] = states
            encodings = lstm_outputs[self.n_r_layers - 1].transpose(1,2)
        else:
            layer_output = preliminary_encodings
            encodings = layer_output
        
        decode_index = self.n_r_layers
        if self.disableRecDeConv: decode_index = 0
        reconstructions = nn.Sequential(*self.decoder_layers[decode_index:])(layer_output)
        reconstructions = reconstructions.transpose(1,2)
        return reconstructions, encodings

# Predictive Sequence2Sequence Recurrent Convs
class CRNN_AE_Seq2Seq(nn.Module):
    def __init__(
        self,
        image_size = 128, 
        channels = 3,
        filter_count = [64]*3 + [96]*2,
        filter_sizes = [3]*5,
        filter_strides = [2]*4 + [1],
        n_r_layers = 2,
        disableRecDeConv = False,
        useBias = False
    ):
        super(CRNN_AE_Seq2Seq, self).__init__()
        self.__name__ = "CRNN_AE_%d_%dx%d_SEQ2SEQ_L-%d_RL-%d_DisDeConv-%s-"%(image_size, most_common(filter_sizes), most_common(filter_sizes), len(filter_count), n_r_layers, "Y" if disableRecDeConv else "N")
        self.image_size = image_size
        self.channels = channels
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableRecDeConv = disableRecDeConv
        
        self.n_layers = len(self.filter_count)
        self.n_r_layers = n_r_layers
        self.n_normal = self.n_layers - self.n_r_layers
        
        assert n_r_layers > 0, "There should be at least one RNN layer"
        assert self.filter_count[-1] == self.filter_count[-2], "Last two filter counts should be same"
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_r_layers:
                insert = TimeDistributed(C2D_BN_A(in_channels, n, k, s))
            else:
                insert = ConvRNN_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias)
            self.encoder_layers.append(insert)
            current_input_shape = getConvOutputShape(current_input_shape, k, s)
            in_channels = n
            
        self.decoder_layers = list()
        
        for idx, (n, k, s) in enumerate(zip(self.filter_count[::-1], self.filter_sizes[::-1], self.filter_strides[::-1])):
            oc_idx = len(self.filter_count) - (2 + idx)
            activation_type = "leaky_relu"
            if oc_idx > -1: out_channels = self.filter_count[oc_idx]
            else:
#                 out_channels = self.channels
                if k%2 !=0: k *= 2
#                 activation_type = "sigmoid"
            if idx < self.n_r_layers and not self.disableRecDeConv:
                insert = ConvTransposeRNN_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
            
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)
        self.conv3d = C3D_BN_A(self.filter_count[0], self.channels, (1,self.filter_sizes[-1],self.filter_sizes[-1]), 1, activation_type = "sigmoid")
        self.adjust_pool = nn.AvgPool2d(3, 1)
    
    def forward(self, x, future_steps = 0):
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_r_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h

        preliminary_encodings = preliminary_encodings.permute(1,0,2,3,4) # ts,[bs,c,w,h]
        if future_steps == 0: future_steps = ts
        
        # Encoding
        encoder_states = [None]*self.n_r_layers
        current_input = preliminary_encodings
        encoder_outputs = list()
        rnn_layers = self.encoder_layers[self.n_normal:]
        
        for idx, layer in enumerate(rnn_layers):
            layer_outputs = list()
            states = encoder_states[idx]
            for t in range(ts):
                y, states = layer(current_input[t], states)
                layer_outputs.append(y)
            layer_outputs = torch.stack(layer_outputs)
            encoder_outputs.append(layer_outputs)
            current_input = layer_outputs
            encoder_states[idx] = states
            
        encodings = y
        
        # Decoder
        current_input = encodings
        decoder_states = [None]*self.n_r_layers
        decoder_outputs = list()
        rnn_layers = self.decoder_layers[:self.n_r_layers]
        
        for t in range(future_steps):
            for idx, layer in enumerate(rnn_layers):
                y, states = layer(current_input, decoder_states[idx])
                decoder_states[idx] = states
                current_input = y
                if idx == 0:
                    next_ts_input = current_input
            current_input = self.adjust_pool(next_ts_input)
            decoder_outputs.append(y)
        
        decoder_outputs = torch.stack(decoder_outputs) # ts,bs,c,w,h
        decoder_outputs = decoder_outputs.permute(1,0,2,3,4) # ts,bs,c,w,h -> bs,ts,c,w,h
        decoder_outputs = nn.Sequential(*self.decoder_layers[self.n_r_layers:])(decoder_outputs) # bs,ts,c,w,h
        reconstructions = self.conv3d(decoder_outputs.permute(0,2,1,3,4)) # bs,ts,c,w,h -> bs,c,ts,w,h
        return reconstructions, encodings

class CGRU_AE_Seq2Seq(nn.Module):
    def __init__(
        self,
        image_size = 128, 
        channels = 3,
        filter_count = [64]*3 + [96]*2,
        filter_sizes = [3]*5,
        filter_strides = [2]*4 + [1],
        n_r_layers = 2,
        disableRecDeConv = False,
        useBias = False
    ):
        super(CGRU_AE_Seq2Seq, self).__init__()
        self.__name__ = "CGRU_AE_%d_%dx%d_SEQ2SEQ_L-%d_RL-%d_DisDeConv-%s-"%(image_size, most_common(filter_sizes), most_common(filter_sizes), len(filter_count), n_r_layers, "Y" if disableRecDeConv else "N")
        self.image_size = image_size
        self.channels = channels
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableRecDeConv = disableRecDeConv
        
        self.n_layers = len(self.filter_count)
        self.n_r_layers = n_r_layers
        self.n_normal = self.n_layers - self.n_r_layers
        
        assert n_r_layers > 0, "There should be at least one GRU layer"
        assert self.filter_count[-1] == self.filter_count[-2], "Last two filter counts should be same"
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_r_layers:
                insert = TimeDistributed(C2D_BN_A(in_channels, n, k, s))
            else:
                insert = ConvGRU_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias)
            self.encoder_layers.append(insert)
            current_input_shape = getConvOutputShape(current_input_shape, k, s)
            in_channels = n
            
        self.decoder_layers = list()
        
        for idx, (n, k, s) in enumerate(zip(self.filter_count[::-1], self.filter_sizes[::-1], self.filter_strides[::-1])):
            oc_idx = len(self.filter_count) - (2 + idx)
            activation_type = "leaky_relu"
            if oc_idx > -1: out_channels = self.filter_count[oc_idx]
            else:
#                 out_channels = self.channels
                if k%2 !=0: k *= 2
#                 activation_type = "sigmoid"
            if idx < self.n_r_layers and not self.disableRecDeConv:
                insert = ConvTransposeGRU_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
            
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)
        self.conv3d = C3D_BN_A(self.filter_count[0], self.channels, (1,self.filter_sizes[-1],self.filter_sizes[-1]), 1, activation_type = "sigmoid")
        self.adjust_pool = nn.AvgPool2d(3, 1)
    
    def forward(self, x, future_steps = 0):
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_r_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h

        preliminary_encodings = preliminary_encodings.permute(1,0,2,3,4) # ts,[bs,c,w,h]
        if future_steps == 0: future_steps = ts
        
        # Encoding
        encoder_states = [None]*self.n_r_layers
        current_input = preliminary_encodings
        encoder_outputs = list()
        gru_layers = self.encoder_layers[self.n_normal:]
        
        for idx, layer in enumerate(gru_layers):
            layer_outputs = list()
            states = encoder_states[idx]
            for t in range(ts):
                y, states = layer(current_input[t], states)
                layer_outputs.append(y)
            layer_outputs = torch.stack(layer_outputs)
            encoder_outputs.append(layer_outputs)
            current_input = layer_outputs
            encoder_states[idx] = states
            
        encodings = y
        
        # Decoder
        current_input = encodings
        decoder_states = [None]*self.n_r_layers
        decoder_outputs = list()
        gru_layers = self.decoder_layers[:self.n_r_layers]
        
        for t in range(future_steps):
            for idx, layer in enumerate(gru_layers):
                y, states = layer(current_input, decoder_states[idx])
                decoder_states[idx] = states
                current_input = y
                if idx == 0:
                    next_ts_input = current_input
            current_input = self.adjust_pool(next_ts_input)
            decoder_outputs.append(y)
        
        decoder_outputs = torch.stack(decoder_outputs) # ts,bs,c,w,h
        decoder_outputs = decoder_outputs.permute(1,0,2,3,4) # ts,bs,c,w,h -> bs,ts,c,w,h
        decoder_outputs = nn.Sequential(*self.decoder_layers[self.n_r_layers:])(decoder_outputs) # bs,ts,c,w,h
        reconstructions = self.conv3d(decoder_outputs.permute(0,2,1,3,4)) # bs,ts,c,w,h -> bs,c,ts,w,h
        return reconstructions, encodings

class CLSTM_AE_Seq2Seq(nn.Module):
    def __init__(
        self,
        image_size = 128, 
        channels = 3,
        filter_count = [64]*3 + [96]*2,
        filter_sizes = [3]*5,
        filter_strides = [2]*4 + [1],
        n_r_layers = 2,
        disableRecDeConv = False,
        useBias = False
    ):
        super(CLSTM_AE_Seq2Seq, self).__init__()
        self.__name__ = "CLSTM_AE_%d_%dx%d_SEQ2SEQ_L-%d_RL-%d_DisDeConv-%s-"%(image_size, most_common(filter_sizes), most_common(filter_sizes), len(filter_count), n_r_layers, "Y" if disableRecDeConv else "N")
        self.image_size = image_size
        self.channels = channels
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableRecDeConv = disableRecDeConv
        
        self.n_layers = len(self.filter_count)
        self.n_r_layers = n_r_layers
        self.n_normal = self.n_layers - self.n_r_layers
        
        assert n_r_layers > 0, "There should be at least one LSTM layer"
        assert self.filter_count[-1] == self.filter_count[-2], "Last two filter counts should be same"
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_r_layers:
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
#                 out_channels = self.channels
                if k%2 !=0: k *= 2
#                 activation_type = "sigmoid"
            if idx < self.n_r_layers and not self.disableRecDeConv:
                insert = ConvTransposeLSTM_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
            
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)
        self.conv3d = C3D_BN_A(self.filter_count[0], self.channels, (1,self.filter_sizes[-1],self.filter_sizes[-1]), 1, activation_type = "sigmoid")
        self.adjust_pool = nn.AvgPool2d(3, 1)
    
    def forward(self, x, future_steps = 0):
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_r_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h

        preliminary_encodings = preliminary_encodings.permute(1,0,2,3,4) # ts,[bs,c,w,h]
        if future_steps == 0: future_steps = ts
        
        # Encoding
        encoder_states = [None]*self.n_r_layers
        current_input = preliminary_encodings
        encoder_outputs = list()
        lstm_layers = self.encoder_layers[self.n_normal:]
        
        for idx, layer in enumerate(lstm_layers):
            layer_outputs = list()
            states = encoder_states[idx]
            for t in range(ts):
                states = layer(current_input[t], states)
                layer_outputs.append(states[0])
            layer_outputs = torch.stack(layer_outputs)
            encoder_outputs.append(layer_outputs)
            current_input = layer_outputs
            encoder_states[idx] = states
            
        encodings = states[0]
        
        # Decoder
        current_input = encodings
        decoder_states = [None]*self.n_r_layers
        decoder_outputs = list()
        lstm_layers = self.decoder_layers[:self.n_r_layers]
        
        for t in range(future_steps):
            for idx, layer in enumerate(lstm_layers):
                states = layer(current_input, decoder_states[idx])
                decoder_states[idx] = states
                current_input = states[0]
                if idx == 0:
                    next_ts_input = current_input
            current_input = self.adjust_pool(next_ts_input)
            decoder_outputs.append(states[0])
        
        decoder_outputs = torch.stack(decoder_outputs) # ts,bs,c,w,h
        decoder_outputs = decoder_outputs.permute(1,0,2,3,4) # ts,bs,c,w,h -> bs,ts,c,w,h
        decoder_outputs = nn.Sequential(*self.decoder_layers[self.n_r_layers:])(decoder_outputs) # bs,ts,c,w,h
        reconstructions = self.conv3d(decoder_outputs.permute(0,2,1,3,4)) # bs,ts,c,w,h -> bs,c,ts,w,h
        return reconstructions, encodings
    
# Attention Recurrent Conv
class CRNN_AE_ATTN(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filter_count = [64]*3 + [96]*2,
        filter_sizes =[3]*5,
        filter_strides = [2]*4 + [1],
        n_r_layers = 2,
        disableRecDeConv = True,
        useBias = False
    ):
        super(CRNN_AE_ATTN, self).__init__()
        self.__name__ = "CRNN_AE_ATTN_%d_%dx%d_L-%d_RL-%d_DisDeConv-%s-"%(image_size, most_common(filter_sizes), most_common(filter_sizes), len(filter_count), n_r_layers, "Y" if disableRecDeConv else "N")
        self.channels = channels
        self.image_size = image_size
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableRecDeConv = disableRecDeConv
        
        self.n_layers = len(self.filter_count)
        self.n_r_layers = n_r_layers
        self.n_normal = self.n_layers - self.n_r_layers
        
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_r_layers:
                insert = TimeDistributed(C2D_BN_A(in_channels, n, k, s))
            else:
                insert = ConvRNN_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias)
            self.encoder_layers.append(insert)
            current_input_shape = getConvOutputShape(current_input_shape, k, s)
            in_channels = n
            
        self.embedding_shape = [1, n, current_input_shape, current_input_shape]
            
        self.decoder_layers = list()
        
        for idx, (n, k, s) in enumerate(zip(self.filter_count[::-1], self.filter_sizes[::-1], self.filter_strides[::-1])):
            oc_idx = len(self.filter_count) - (2 + idx)
            activation_type = "leaky_relu"
            if oc_idx > -1: out_channels = self.filter_count[oc_idx]
            else:
                out_channels = self.channels
                if k%2 !=0: k += 1
                activation_type = "sigmoid"
            if idx < self.n_r_layers and not self.disableRecDeConv:
                insert = ConvTransposeRNN_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
        
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)
        self.attention = TimeDistributed(LinearAttentionLayer(np.product(self.embedding_shape)))
                                
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_r_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h
        
        if self.n_r_layers != 0:
            states_list = [None] * 2 * self.n_r_layers
            current_input = preliminary_encodings
            rnn_outputs = list()
            rnn_layers = self.encoder_layers[self.n_normal:]
            if not self.disableRecDeConv: rnn_layers += self.decoder_layers[:self.n_r_layers]
            for idx, layer in enumerate(rnn_layers):
                layer_outputs = list()
                states = states_list[idx]
                for t in range(ts):
                    y, h = layer(current_input[:,t,...], states)
                    layer_outputs.append(y)
                    states = h
                layer_output = torch.stack(layer_outputs, dim = 1) # b,ts,c,w,h
                if idx == (self.n_r_layers-1):
                    enc = layer_output
                    enc = self.attention(enc.flatten(start_dim=2))
                    layer_output = enc.reshape(*layer_output.shape)
                rnn_outputs.append(layer_output)
                current_input = layer_output
                states_list[idx] = states
            encodings = rnn_outputs[self.n_r_layers - 1].transpose(1,2)
        else:
            layer_output = preliminary_encodings
            encodings = layer_output
        
        decode_index = self.n_r_layers
        if self.disableRecDeConv: decode_index = 0
        reconstructions = nn.Sequential(*self.decoder_layers[decode_index:])(layer_output)
        reconstructions = reconstructions.transpose(1,2)
        return reconstructions, encodings

class CRNN_AE_ATTN2(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filter_count = [64]*3 + [96]*2,
        filter_sizes =[3]*5,
        filter_strides = [2]*4 + [1],
        n_r_layers = 2,
        disableRecDeConv = True,
        useBias = False
    ):
        super(CRNN_AE_ATTN2, self).__init__()
        self.__name__ = "CRNN_AE_ATTN2_%d_%dx%d_L-%d_RL-%d_DisDeConv-%s-"%(image_size, most_common(filter_sizes), most_common(filter_sizes), len(filter_count), n_r_layers, "Y" if disableRecDeConv else "N")
        self.channels = channels
        self.image_size = image_size
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableRecDeConv = disableRecDeConv
        
        self.n_layers = len(self.filter_count)
        self.n_r_layers = n_r_layers
        self.n_normal = self.n_layers - self.n_r_layers
        
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_r_layers:
                insert = TimeDistributed(C2D_BN_A(in_channels, n, k, s))
            else:
                insert = ConvRNN_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias)
            self.encoder_layers.append(insert)
            current_input_shape = getConvOutputShape(current_input_shape, k, s)
            in_channels = n
            
        self.embedding_shape = [1, n, current_input_shape, current_input_shape]
            
        self.decoder_layers = list()
        
        for idx, (n, k, s) in enumerate(zip(self.filter_count[::-1], self.filter_sizes[::-1], self.filter_strides[::-1])):
            oc_idx = len(self.filter_count) - (2 + idx)
            activation_type = "leaky_relu"
            if oc_idx > -1: out_channels = self.filter_count[oc_idx]
            else:
                out_channels = self.channels
                if k%2 !=0: k += 1
                activation_type = "sigmoid"
            if idx < self.n_r_layers and not self.disableRecDeConv:
                insert = ConvTransposeRNN_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
        
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)
        self.attention = TimeDistributed(SoftMaxConvAttentionRNNWrapper(3))
                                
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        # bs,c,ts,w,h -> bs,ts,c,w,h -> attention -> bs,ts,c,w,h -> bs,c,ts,w,h
        x = self.attention(x.permute(0,2,1,3,4)).permute(0,2,1,3,4)
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_r_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h
        
        if self.n_r_layers != 0:
            states_list = [None] * 2 * self.n_r_layers
            current_input = preliminary_encodings
            rnn_outputs = list()
            rnn_layers = self.encoder_layers[self.n_normal:]
            if not self.disableRecDeConv: rnn_layers += self.decoder_layers[:self.n_r_layers]
            for idx, layer in enumerate(rnn_layers):
                layer_outputs = list()
                states = states_list[idx]
                for t in range(ts):
                    y, h = layer(current_input[:,t,...], states)
                    layer_outputs.append(y)
                    states = h
                layer_output = torch.stack(layer_outputs, dim = 1) # b,ts,c,w,h
                rnn_outputs.append(layer_output)
                current_input = layer_output
                states_list[idx] = states
            encodings = rnn_outputs[self.n_r_layers - 1].transpose(1,2)
        else:
            layer_output = preliminary_encodings
            encodings = layer_output
        
        decode_index = self.n_r_layers
        if self.disableRecDeConv: decode_index = 0
        reconstructions = nn.Sequential(*self.decoder_layers[decode_index:])(layer_output)
        reconstructions = reconstructions.transpose(1,2)
        return reconstructions, encodings
    
# Bidirectional Recurrent Convs
class BiCRNN_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filter_count = [64]*3 + [96]*2,
        filter_sizes =[3]*5,
        filter_strides = [2]*4 + [1],
        n_r_layers = 2,
        disableRecDeConv = True,
        useBias = False
    ):
        super(BiCRNN_AE, self).__init__()
        self.__name__ = "BiCRNN_AE_%d_%dx%d_L-%d_RL-%d_DisDeConv-%s-"%(image_size, most_common(filter_sizes), most_common(filter_sizes), len(filter_count), n_r_layers, "Y" if disableRecDeConv else "N")
        self.channels = channels
        self.image_size = image_size
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableRecDeConv = disableRecDeConv
        
        self.n_layers = len(self.filter_count)
        self.n_r_layers = n_r_layers
        self.n_normal = self.n_layers - self.n_r_layers
        
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_r_layers:
                insert = TimeDistributed(C2D_BN_A(in_channels, n, k, s))
            else:
                insert = nn.ModuleList([
                    ConvRNN_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias),
                    ConvRNN_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias)
                ])
                
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
            if idx < self.n_r_layers and not self.disableRecDeConv:
                insert = nn.ModuleList([
                    ConvTransposeRNN_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias),
                    ConvTransposeRNN_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
                ])
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
        
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)

    def rnn_forward(self, x, isBackward = False):
        bs,ts,c,w,h = x.shape
        
        rnn_layer_idx = 0
        if isBackward: rnn_layer_idx = 1
        # x -> bs,ts,c,w,h
        
        if self.n_r_layers != 0:
            states_list = [None] * 2 * self.n_r_layers
            current_input = x
            rnn_outputs = list()
            rnn_layers = self.encoder_layers[self.n_normal:]
            if not self.disableRecDeConv: rnn_layers += self.decoder_layers[:self.n_r_layers]
            for idx, layer in enumerate(rnn_layers):
                layer_outputs = list()
                states = states_list[idx]
                for t in range(ts):
                    y, h = layer[rnn_layer_idx](current_input[:,t,...], states)
                    layer_outputs.append(y)
                    states = h
                layer_output = torch.stack(layer_outputs, dim = 1) # b,ts,c,w,h
                rnn_outputs.append(layer_output)
                current_input = layer_output
                states_list[idx] = states
            encodings = rnn_outputs[self.n_r_layers - 1].transpose(1,2)
        else:
            layer_output = x
            encodings = layer_output
        return layer_output, encodings
    
    def flip_temporal(self, x):
        # temporal_idx = 1
        return torch.fliplr(x.transpose(0,1)).transpose(0,1)
        
    def forward(self, x):
    
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_r_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h
        
        f_layer_output, f_encodings = self.rnn_forward(preliminary_encodings)
        b_layer_output, b_encodings = self.rnn_forward(self.flip_temporal(preliminary_encodings), isBackward = True)
        
        layer_output = torch.mean(torch.stack([f_layer_output, self.flip_temporal(b_layer_output)]), dim = 0) #.squeeze(dim=0)
        encodings = torch.mean(torch.stack([f_encodings, self.flip_temporal(b_encodings)]), dim = 0) #.squeeze(dim=0)
        
        decode_index = self.n_r_layers
        if self.disableRecDeConv: decode_index = 0
        reconstructions = nn.Sequential(*self.decoder_layers[decode_index:])(layer_output)
        reconstructions = reconstructions.transpose(1,2)
        return reconstructions, encodings
    
class BiCGRU_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filter_count = [64]*3 + [96]*2,
        filter_sizes =[3]*5,
        filter_strides = [2]*4 + [1],
        n_r_layers = 2,
        disableRecDeConv = True,
        useBias = False
    ):
        super(BiCGRU_AE, self).__init__()
        self.__name__ = "BiCGRU_AE_%d_%dx%d_L-%d_RL-%d_DisDeConv-%s-"%(image_size, most_common(filter_sizes), most_common(filter_sizes), len(filter_count), n_r_layers, "Y" if disableRecDeConv else "N")
        self.channels = channels
        self.image_size = image_size
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableRecDeConv = disableRecDeConv
        
        self.n_layers = len(self.filter_count)
        self.n_r_layers = n_r_layers
        self.n_normal = self.n_layers - self.n_r_layers
        
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_r_layers:
                insert = TimeDistributed(C2D_BN_A(in_channels, n, k, s))
            else:
                insert = nn.ModuleList([
                    ConvGRU_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias),
                    ConvGRU_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias)
                ])
                
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
            if idx < self.n_r_layers and not self.disableRecDeConv:
                insert = nn.ModuleList([
                    ConvTransposeGRU_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias),
                    ConvTransposeGRU_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
                ])
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
        
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)

    def gru_forward(self, x, isBackward = False):
        bs,ts,c,w,h = x.shape
        
        gru_layer_idx = 0
        if isBackward: gru_layer_idx = 1
        # x -> bs,ts,c,w,h
        
        if self.n_r_layers != 0:
            states_list = [None] * 2 * self.n_r_layers
            current_input = x
            gru_outputs = list()
            gru_layers = self.encoder_layers[self.n_normal:]
            if not self.disableRecDeConv: gru_layers += self.decoder_layers[:self.n_r_layers]
            for idx, layer in enumerate(gru_layers):
                layer_outputs = list()
                states = states_list[idx]
                for t in range(ts):
                    y, h = layer[gru_layer_idx](current_input[:,t,...], states)
                    layer_outputs.append(y)
                    states = h
                layer_output = torch.stack(layer_outputs, dim = 1) # b,ts,c,w,h
                gru_outputs.append(layer_output)
                current_input = layer_output
                states_list[idx] = states
            encodings = gru_outputs[self.n_r_layers - 1].transpose(1,2)
        else:
            layer_output = x
            encodings = layer_output
        return layer_output, encodings
    
    def flip_temporal(self, x):
        # temporal_idx = 1
        return torch.fliplr(x.transpose(0,1)).transpose(0,1)
        
    def forward(self, x):
    
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_r_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h
        
        f_layer_output, f_encodings = self.gru_forward(preliminary_encodings)
        b_layer_output, b_encodings = self.gru_forward(self.flip_temporal(preliminary_encodings), isBackward = True)
        
        layer_output = torch.mean(torch.stack([f_layer_output, self.flip_temporal(b_layer_output)]), dim = 0) #.squeeze(dim=0)
        encodings = torch.mean(torch.stack([f_encodings, self.flip_temporal(b_encodings)]), dim = 0) #.squeeze(dim=0)
        
        decode_index = self.n_r_layers
        if self.disableRecDeConv: decode_index = 0
        reconstructions = nn.Sequential(*self.decoder_layers[decode_index:])(layer_output)
        reconstructions = reconstructions.transpose(1,2)
        return reconstructions, encodings
    
class BiCLSTM_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filter_count = [64]*3 + [96]*2,
        filter_sizes =[3]*5,
        filter_strides = [2]*4 + [1],
        n_r_layers = 2,
        disableRecDeConv = True,
        useBias = False
    ):
        super(BiCLSTM_AE, self).__init__()
        self.__name__ = "BiCLSTM_AE_%d_%dx%d_L-%d_RL-%d_DisDeConv-%s-"%(image_size, most_common(filter_sizes), most_common(filter_sizes), len(filter_count), n_r_layers, "Y" if disableRecDeConv else "N")
        self.channels = channels
        self.image_size = image_size
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableRecDeConv = disableRecDeConv
        
        self.n_layers = len(self.filter_count)
        self.n_r_layers = n_r_layers
        self.n_normal = self.n_layers - self.n_r_layers
        
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_r_layers:
                insert = TimeDistributed(C2D_BN_A(in_channels, n, k, s))
            else:
                insert = nn.ModuleList([
                    ConvLSTM_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias),
                    ConvLSTM_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias)
                ])
                
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
            if idx < self.n_r_layers and not self.disableRecDeConv:
                insert = nn.ModuleList([
                    ConvTransposeLSTM_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias),
                    ConvTransposeLSTM_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
                ])
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
        
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)

    def lstm_forward(self, x, isBackward = False):
        bs,ts,c,w,h = x.shape
        
        lstm_layer_idx = 0
        if isBackward: lstm_layer_idx = 1
        # x -> bs,ts,c,w,h
        
        if self.n_r_layers != 0:
            states_list = [None] * 2 * self.n_r_layers
            current_input = x
            lstm_outputs = list()
            lstm_layers = self.encoder_layers[self.n_normal:]
            if not self.disableRecDeConv: lstm_layers += self.decoder_layers[:self.n_r_layers]
            for idx, layer in enumerate(lstm_layers):
                layer_outputs = list()
                states = states_list[idx]
                for t in range(ts):
                    states = layer[lstm_layer_idx](current_input[:,t,...], states)
                    layer_outputs.append(states[0])
                layer_output = torch.stack(layer_outputs, dim = 1) # b,ts,c,w,h
                lstm_outputs.append(layer_output)
                current_input = layer_output
                states_list[idx] = states
            encodings = lstm_outputs[self.n_r_layers - 1].transpose(1,2)
        else:
            layer_output = x
            encodings = layer_output
        return layer_output, encodings
    
    def flip_temporal(self, x):
        # temporal_idx = 1
        return torch.fliplr(x.transpose(0,1)).transpose(0,1)
                                
    def forward(self, x):
            
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_r_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h
        
        f_layer_output, f_encodings = self.lstm_forward(preliminary_encodings)
        b_layer_output, b_encodings = self.lstm_forward(self.flip_temporal(preliminary_encodings), isBackward = True)
        
        layer_output = torch.mean(torch.stack([f_layer_output, self.flip_temporal(b_layer_output)]), dim = 0) #.squeeze(dim=0)
        encodings = torch.mean(torch.stack([f_encodings, self.flip_temporal(b_encodings)]), dim = 0) #.squeeze(dim=0)
        
        decode_index = self.n_r_layers
        if self.disableRecDeConv: decode_index = 0
        reconstructions = nn.Sequential(*self.decoder_layers[decode_index:])(layer_output)
        reconstructions = reconstructions.transpose(1,2)
        return reconstructions, encodings