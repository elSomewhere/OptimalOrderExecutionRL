import torch
import torch.nn.functional as F

'''
note on input shape: we assume 10 levels of LOB depth -> 40 numberes. + 3 agent specific features; total_state_size = 43
'''





class TorchNetwork_DTYPE(torch.nn.Module):
    '''
    So to avoid always needing to specify dtype (pain if switching from cuda to non cuda)
    '''
    def __init__(self, dtype=None):
        super(TorchNetwork_DTYPE, self).__init__()
        if dtype is None or False:
            self.dtype = torch.FloatTensor
        if dtype is True:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = dtype

class TorchNetwork_MODEL(TorchNetwork_DTYPE):
    '''
    Cus feature size changes with changing LOB depth
    '''
    def __init__(self, number_of_features, dtype=None):
        super(TorchNetwork_MODEL, self).__init__(dtype)
        self.number_of_features = number_of_features



class TSM_LSTM_varSize(TorchNetwork_MODEL):
    '''
    Vanilla torch LSTM only supports equally sized multi-layers, this implementation allows for variable-sized layers
    '''
    def __init__(self, number_of_features, hidden_features_per_layer, dtype=None):
        super(TSM_LSTM_varSize, self).__init__(number_of_features, dtype)
        self.hidden_features_per_layer = hidden_features_per_layer
        num_levels = len(self.hidden_features_per_layer)
        self.layers = []
        for i in range(num_levels):
            torch.manual_seed(0)
            if i == 0:
                self.layers += [torch.nn.LSTM(
                    input_size=number_of_features,
                    hidden_size=self.hidden_features_per_layer[i],
                    num_layers=1,
                    batch_first=True
                ).type(self.dtype)]
            else:
                self.layers += [
                    torch.nn.LSTM(
                        input_size=hidden_features_per_layer[i - 1],
                        hidden_size=self.hidden_features_per_layer[i],
                        num_layers=1,
                        batch_first=True
                    ).type(self.dtype)]
        self.network = torch.nn.Sequential(*self.layers)

class TSM_LSTM_vanilla(TorchNetwork_MODEL):
    '''
    Baseline LSTM implementation, offers more flexibility than the vanilla torch LSTM cell
    '''
    def __init__(self, number_of_features, hidden_size, num_layers, dropout, dtype=None):
        super(TSM_LSTM_vanilla, self).__init__(number_of_features, dtype)
        torch.manual_seed(0)
        self.lstm = torch.nn.LSTM(
            input_size=number_of_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        ).type(self.dtype)


    def forward(self, input, state_override=None):
        input = input.transpose(1, 2)
        if state_override is not None:
            h_0 = torch.cat([i[0] for i in state_override], 0)
            c_0 = torch.cat([i[1] for i in state_override], 0)
            state_override = (h_0, c_0)
            # state override expects a list of 2d tuples (and not a 3d tensor like in vanilla)
            output, (h_n, c_n) = self.lstm(input, state_override)
        else:
            output, (h_n, c_n) = self.lstm(input)
        h_n = [h_n[i, :, :].unsqueeze(0) for i in range(h_n.shape[0])]
        c_n = [c_n[i, :, :].unsqueeze(0) for i in range(c_n.shape[0])]
        output = output.transpose(1, 2)
        return output, (h_n, c_n)

    def loss(self, reconstruct, target, loss_fn):
        loss = loss_fn(reconstruct, target)
        return loss


class TSM_LSTM(TorchNetwork_MODEL):
    '''
    Final LSTM class which falls back to torch LSTM in case of equal hidden layers since that one has some extra optimizations
    '''
    def __init__(self, number_of_features, hidden_features_per_layer, dropout, dtype=None):
        super(TSM_LSTM, self).__init__(number_of_features, dtype)
        if len(set(hidden_features_per_layer)) == 1:
            self.lstm_type = "vanilla"
            self.lstm = TSM_LSTM_vanilla(number_of_features, set(hidden_features_per_layer).pop(), len(hidden_features_per_layer),
                                         dropout, dtype)
        else:
            print("Warning: variable hidden sized lstms currently do not support native dropout. They are likely also less cuda optimized than their equal hidden sized counterparts. Recommended to avoid variable sized lstm verison.")
            self.lstm_type = "custom"
            self.lstm = TSM_LSTM_varSize(number_of_features, hidden_features_per_layer, dtype)

    def forward(self, x, state_override=None):
        output, (h_n, c_n) = self.lstm(x, state_override)
        return output, (h_n, c_n)


class RLAgent(TorchNetwork_MODEL):
    def __init__(self, lob_levels, aux_features, time_depth, dtype=None):
        super(RLAgent, self).__init__(number_of_features=lob_levels*4+aux_features, dtype=dtype)
        self.levels = lob_levels
        self.aux_features = aux_features
        self.time_depth = time_depth



class LOB_RNN_DISCRETE(RLAgent):
    '''
    Network with architecture specialized for L2 LOB data.
    Convolutional layers with kernel = 2 & stride 2 transfer [p, v, p, v, p, v...] formatted input LOBs
    into something resembling micro-weighted VWP features at the input stage.
    '''
    def __init__(self,
                 levels,
                 aux_features,
                 time_depth,
                 conv_channels_a,
                 conv_channels_b,
                 conv_channels_c,
                 translation_layer,
                 hidden_features_per_layer,
                 dropout=0.2,
                 dtype=torch.float
                 ):
        super(LOB_RNN_DISCRETE, self).__init__(levels, aux_features, time_depth, dtype)
        self.volumeWeightedPrice = torch.nn.Conv2d(in_channels=1, out_channels=conv_channels_a, kernel_size=(1, 2), stride=(1, 2))
        self.volumeWeightedPrice_act = torch.nn.ReLU()
        self.microPrice = torch.nn.Conv2d(in_channels=conv_channels_a, out_channels=conv_channels_b, kernel_size=(1, 2), stride=(1, 2))
        self.microPrice_act = torch.nn.ReLU()
        self.weightedMidPrice = torch.nn.Conv2d(in_channels=conv_channels_b, out_channels=conv_channels_c, kernel_size=(1, levels), stride=(1, levels))
        self.weightedMidPrice_act = torch.nn.ReLU()
        self.totalstate_to_rnn = torch.nn.Linear(conv_channels_c + self.aux_features, translation_layer).type(self.dtype)
        self.totalstate_to_rnn_act = torch.nn.ReLU()
        self.encoder = TSM_LSTM(
            number_of_features=translation_layer,
            hidden_features_per_layer=hidden_features_per_layer,
            dropout=dropout,
            dtype=dtype
        )
        self.expand_output_mean = torch.nn.Linear(hidden_features_per_layer[-1], 1).type(self.dtype)
        # self.expand_output_std = torch.nn.Linear(hidden_features_per_layer[-1], 1).type(self.dtype)
        self.log_std = torch.nn.Parameter(torch.zeros(1))  # Learnable log standard deviation

    # input needs an extra dimension because conv2 always asumes a layer channel
    def forward(self, x):
        '''
        Expects shape: [batchsize, statesize, timesteps]
        '''
        # add a convolution dimension
        x = x.unsqueeze(1)
        # split lob and agent state, they will be processed seaparately in the feature layers
        lob = x[:, :, 0:(self.levels * 4), :]
        aux = x[:, :, -self.aux_features::, :]
        # create volume weighted midprice
        x_vwp = self.volumeWeightedPrice_act(self.volumeWeightedPrice(lob.transpose(2, 3)))
        # create microprice
        x_mp = self.microPrice_act(self.microPrice(x_vwp))
        # create weighted midprice
        x_weightedMidPrice = self.weightedMidPrice_act(self.weightedMidPrice(x_mp)).squeeze(3)
        x_concatenated = torch.cat([x_weightedMidPrice, aux.squeeze(1)], 1)
        # combine midprice and agent state
        x_final_to_rnn = self.totalstate_to_rnn_act(self.totalstate_to_rnn(x_concatenated.transpose(1, 2)).transpose(1, 2))
        # flatten time
        output, all_hidden_states = self.encoder(x_final_to_rnn)
        # get output of RNN
        prediction_mean = self.expand_output_mean(output[:, :, -1])
        # prediction_std = self.expand_output_std(output[:, :, -1])
        final_output_mean = torch.tanh(prediction_mean).squeeze(1)
        # final_output_std = torch.exp(prediction_std).squeeze(1)
        final_output_std = torch.exp(self.log_std)  # Standard deviation
        print("mean: "+str(final_output_mean)+" std: "+str(final_output_std))
        return final_output_mean, final_output_std