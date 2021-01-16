from .recurrent_convs import *

class CRNN_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filter_count = [64,64,128,128],
        filter_sizes = [3,3,3,5],
        filter_strides = [2,2,2,2],
        n_rnn_layers = 2,
        disableDeConvRNN = True,
        useBias = False
    ):
        super(CRNN_AE, self).__init__()
        self.__name__ = "CRNN_AE_v2_%d"%(image_size)
        self.channels = channels
        self.image_size = image_size
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableDeConvRNN = disableDeConvRNN
        
        self.n_layers = len(self.filter_count)
        self.n_rnn_layers = n_rnn_layers
        self.n_normal = self.n_layers - self.n_rnn_layers
        
        assert len(filter_count) == len(filter_sizes), "Number of filter sizes and count should be the same"
        assert len(filter_count) == len(filter_strides), "Number of filter strides and count should be the same"
        
        current_input_shape = self.image_size
        in_channels = self.channels
        
        self.encoder_layers = list()
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            if (self.n_layers - idx) > self.n_rnn_layers:
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
            if idx < self.n_rnn_layers and not self.disableDeConvRNN:
                insert = ConvTransposeRNN_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
        
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)
                                
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_rnn_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h
        
        if self.n_rnn_layers != 0:
            states_list = [None] * 2 * self.n_rnn_layers
            current_input = preliminary_encodings
            rnn_outputs = list()
            rnn_layers = self.encoder_layers[self.n_normal:]
            if not self.disableDeConvRNN: rnn_layers += self.decoder_layers[:self.n_rnn_layers]
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
            encodings = rnn_outputs[self.n_rnn_layers - 1].transpose(1,2)
        else:
            layer_output = preliminary_encodings
            encodings = layer_output
        
        decode_index = self.n_rnn_layers
        if self.disableDeConvRNN: decode_index = 0
        reconstructions = nn.Sequential(*self.decoder_layers[decode_index:])(layer_output)
        reconstructions = reconstructions.transpose(1,2)
        return reconstructions, encodings
    
class CLSTM_AE(nn.Module):
    def __init__(
        self,
        image_size = 128,
        channels = 3,
        filter_count = [64,64,128,128],
        filter_sizes = [3,3,3,5],
        filter_strides = [2,2,2,2],
        n_lstm_layers = 2,
        disableDeConvLSTM = True,
        useBias = False
    ):
        super(CLSTM_AE, self).__init__()
        self.__name__ = "CLSTM_AE_v2_%d"%(image_size)
        self.channels = channels
        self.image_size = image_size
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        self.disableDeConvLSTM = disableDeConvLSTM
        
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
            if idx < self.n_lstm_layers and not self.disableDeConvLSTM:
                insert = ConvTransposeLSTM_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
            else:
                insert = TimeDistributed(CT2D_BN_A(n, out_channels, k, s, activation_type = activation_type))
            self.decoder_layers.append(insert)
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
        
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)
                                
    def forward(self, x):
        bs,c,ts,w,h = x.shape
        preliminary_encodings = nn.Sequential(*self.encoder_layers[:(self.n_layers - self.n_lstm_layers)])(x.permute(0,2,1,3,4)) # bs,ts,c,w,h
        # preliminary_encodings -> bs,ts,c,w,h
        
        if self.n_lstm_layers != 0:
            states_list = [None] * 2 * self.n_lstm_layers
            current_input = preliminary_encodings
            lstm_outputs = list()
            lstm_layers = self.encoder_layers[self.n_normal:]
            if not self.disableDeConvLSTM: lstm_layers += self.decoder_layers[:self.n_lstm_layers]
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
            encodings = lstm_outputs[self.n_lstm_layers - 1].transpose(1,2)
        else:
            layer_output = preliminary_encodings
            encodings = layer_output
        
        decode_index = self.n_lstm_layers
        if self.disableDeConvLSTM: decode_index = 0
        reconstructions = nn.Sequential(*self.decoder_layers[decode_index:])(layer_output)
        reconstructions = reconstructions.transpose(1,2)
        return reconstructions, encodings
    
class CLSTM_Seq2Seq(nn.Module):
    def __init__(
        self,
        image_size = 128, 
        channels = 3,
        filter_count = [64,64,64],
        filter_sizes = [3,3,3],
        filter_strides = [2,2,1],
        useBias = False
    ):
        super(CLSTM_Seq2Seq, self).__init__()
        self.__name__ = "CLSTM_Seq2Seq_%d"%(image_size)
        self.image_size = image_size
        self.channels = channels
        self.filter_count = filter_count
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides
        
        assert self.filter_count[-1] == self.filter_count[-2], "Last two filter counts should be same"
        
        self.encoder_layers = list()
        current_input_shape = self.image_size
        in_channels = self.channels
        for idx, (n, k, s) in enumerate(zip(self.filter_count, self.filter_sizes, self.filter_strides)):
            self.encoder_layers.append(
                ConvLSTM_Cell(current_input_shape, in_channels, n, k, s, useBias=useBias)
            )
            current_input_shape = getConvOutputShape(current_input_shape,k,s)
            in_channels = n
            
        self.decoder_layers = list()
        
        for idx, (n, k, s) in enumerate(zip(self.filter_count[::-1], self.filter_sizes[::-1], self.filter_strides[::-1])):
            oc_idx = len(self.filter_count) - (2 + idx)
            activation_type = "leaky_relu"
            if oc_idx > -1: out_channels = self.filter_count[oc_idx]
            else:
                states = None
                if k%2 !=0: k *= 2
            self.decoder_layers.append(
                ConvTransposeLSTM_Cell(current_input_shape, n, out_channels, k, s, useBias=useBias)
            )
            current_input_shape = getConvTransposeOutputShape(current_input_shape, k, s)
        
        # For cuda use
        self.modules = nn.ModuleList(self.encoder_layers + self.decoder_layers)
        self.conv3d = C3D_BN_A(self.filter_count[0], self.channels, (1,self.filter_sizes[-1],self.filter_sizes[-1]), 1, activation_type = "sigmoid")
        self.adjust_pool = nn.AvgPool2d(3, 1)
    
    def forward(self, x, future_steps = 0):
        bs,c,ts,w,h = x.shape
        x = x.permute(2,0,1,3,4) # ts,[bs,c,w,h]
        if future_steps == 0: future_steps = ts
        
        # Encoding
        encoder_states = [None]*len(self.encoder_layers)
        current_input = x
        encoder_outputs = list()
        for idx, layer in enumerate(self.encoder_layers):
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
        decoder_states = [None]*len(self.decoder_layers)
        decoder_outputs = list()
        
        for t in range(future_steps):
            for idx, layer in enumerate(self.decoder_layers):
                states = layer(current_input, decoder_states[idx])
                decoder_states[idx] = states
                current_input = states[0]
                if idx == 0:
                    next_ts_input = current_input
            current_input = self.adjust_pool(next_ts_input)
            decoder_outputs.append(states[0])
        
        decoder_outputs = torch.stack(decoder_outputs).permute(1,2,0,3,4) # ts,bs,c,w,h -> bs,c,ts,w,h
        reconstructions = self.conv3d(decoder_outputs)
        return reconstructions, encodings