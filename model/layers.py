import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DNorm(nn.Module):
    def __init__(self, input_size, output_size, kernel, relu=True):
        super(Conv1DNorm, self).__init__()
        padding = kernel//2
        layers = [nn.Conv1d(input_size, output_size, kernel, stride=1, padding=padding, bias=False)]
        if relu: 
            layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(output_size))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class Highway(nn.Module):
    def __init__(self, input_size, output_size):
        super(Highway, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear1.bias.data.zero_()
        self.linear2 = nn.Linear(input_size, output_size)
        self.linear2.bias.data.fill_(-1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.relu(self.linear1(x))
        h2 = self.sigmoid(self.linear2(x))

        return h1*h2 + x*(1.0 - h2)

class Prenet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Prenet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        return self.layers(x)

class CBHG(nn.Module):
    def __init__(self, input_size, projections, K, num_highways):
        super(CBHG, self).__init__()
        # Conv1D Banks 
        self.kernels = [i for i in range(1, K + 1)]
        self.conv1d_banks = nn.ModuleList()
        for k in self.kernels:
            conv = Conv1DNorm(input_size, input_size, k)
            self.conv1d_banks.append(conv)

        # Max pooling && Conv1D Projections
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.conv_projection = nn.Sequential(
            Conv1DNorm(len(self.kernels)*input_size, projections[0], 3),
            Conv1DNorm(projections[0], projections[1], 3, relu=False))

        # Highway Layers
        highways = []
        if projections[-1] != input_size:
            highways.append(nn.Linear(projections[-1], input_size, bias=False))
        for _ in range(num_highways):
            highways.append(Highway(input_size, input_size))
        self.highways = nn.Sequential(*highways)

        # Bidirectional RNN
        self.rnn = nn.GRU(input_size, input_size, batch_first=True, bidirectional=True)
        self.rnn.flatten_parameters()

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)
        shortcut = x

        # Conv1D Banks 
        conv_bank = []
        for conv in self.conv1d_banks:
            conv_bank.append(conv(x)[:, :, :x.size(-1)])
        conv_bank = torch.cat(conv_bank, dim=1)

        # Max pooling && Conv1D Projections
        x = self.maxpool(conv_bank)[:, :, :x.size(-1)]
        x = self.conv_projection(x)
        x = x + shortcut
        x = x.transpose(1, 2)

        # Highway Layers
        x = self.highways(x)

        # Bidirectional RNN
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
            
        self.rnn.flatten_parameters()
        outputs, _ = self.rnn(x)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs
