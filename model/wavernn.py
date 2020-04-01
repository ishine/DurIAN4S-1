import torch
import torch.nn as nn
import torch.nn.functional as F
from wavernn_util import cumproduct, mol_distribution, decode_mu_law, xfade_and_unfold

class ResBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv1 = nn.Conv1d(size, size, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(size, size, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(size)
        self.batch_norm2 = nn.BatchNorm1d(size)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual

class MelResNet(nn.Module):
    def __init__(self, res_blocks, in_size, compute_size, res_out_size, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_size, compute_size, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_size)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_size))
        self.conv_out = nn.Conv1d(compute_size, res_out_size, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers: x = f(x)
        x = self.conv_out(x)
        return x

class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)

class UpsampleNetwork(nn.Module):
    def __init__(self, feat_size, upsample_scales, compute_size,
                 res_blocks, res_out_size, pad):
        super().__init__()
        total_scale = cumproduct(upsample_scales)[-1]
        self.indent = pad*total_scale
        self.resnet = MelResNet(res_blocks, feat_size, compute_size, res_out_size, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers: m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)

class WaveRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.pad = config.pad
        if self.mode == 'RAW':
            self.n_classes = 2**config.bits
        elif self.mode == 'MOL':
            self.n_classes = 30
        else:
            RuntimeError("Unknown model mode value - ", self.mode)

        # List of rnns to call `flatten_parameters()` on
        self._to_flatten = []

        self.rnn_size = config.rnn_size
        self.aux_size = config.res_out_size//4
        self.hop_size = config.hop_size
        self.sample_rate = config.sample_rate

        self.upsample = UpsampleNetwork(config.mel_size, 
                                        config.upsample_factors, 
                                        config.compute_size, 
                                        config.res_blocks, 
                                        config.res_out_size, 
                                        config.pad)
        self.I = nn.Linear(config.mel_size + self.aux_size + 1, config.rnn_size)

        self.rnn1 = nn.GRU(config.rnn_size, config.rnn_size, batch_first=True)
        self.rnn2 = nn.GRU(config.rnn_size + self.aux_size, config.rnn_size, batch_first=True)
        self._to_flatten += [self.rnn1, self.rnn2]

        self.fc1 = nn.Linear(config.rnn_size + self.aux_size, config.fc_size)
        self.fc2 = nn.Linear(config.fc_size + self.aux_size, config.fc_size)
        self.fc3 = nn.Linear(config.fc_size, self.n_classes)

        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.num_params()

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

    def forward(self, x, mels):
        device = next(self.parameters()).device  # use same device as parameters

        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        self.step += 1
        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.rnn_size, device=device)
        h2 = torch.zeros(1, bsize, self.rnn_size, device=device)
        mels, aux = self.upsample(mels)

        aux_idx = [self.aux_size * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def generate(self, mel, batched, target, overlap, mu_law):
        self.eval()

        mu_law = mu_law if self.mode == 'RAW' else False

        output = []
        # start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():
            wave_len = (mel.size(-1) - 1) * self.hop_size
            mel = self.pad_tensor(mel.transpose(1, 2), pad=self.pad, side='both')
            mel, aux = self.upsample(mel.transpose(1, 2))

            if batched:
                mel = self.fold_with_overlap(mel, target, overlap)
                aux = self.fold_with_overlap(aux, target, overlap)

            b_size, seq_len, _ = mel.size()

            h1 = mel.new_zeros(b_size, self.rnn_size)
            h2 = mel.new_zeros(b_size, self.rnn_size)
            x = mel.new_zeros(b_size, 1)

            d = self.aux_size
            aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

            for i in range(seq_len):

                m_t = mel[:, i, :]

                a1_t, a2_t, a3_t, a4_t = \
                    (a[:, i, :] for a in aux_split)

                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))

                logits = self.fc3(x)

                if self.mode == 'MOL':
                    sample = mol_distribution(logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    # x = torch.FloatTensor([[sample]]).cuda()
                    x = sample.transpose(0, 1)

                elif self.mode == 'RAW':
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)

                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError("Unknown model mode value - ", self.mode)

                # if i % 100 == 0: self.gen_display(i, seq_len, b_size, start)

        output = torch.stack(output).transpose(0, 1).double()
        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)
        if batched:
            output = xfade_and_unfold(output, target, overlap)
        
        output = output[:wave_len]
        '''
        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * self.hop_size)
        output = output[:wave_len]
        output[-20 * self.hop_size:] *= fade_out

        #save_wav(output, save_path)
        '''

        return output

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def pad_tensor(self, x, pad, side='both'):
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/size
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        padded = torch.zeros(b, total, c, device=x.device)
        if side == 'before' or side == 'both':
            padded[:, pad:pad + t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
        return padded

    def fold_with_overlap(self, x, target, overlap):

        ''' Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()

        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)

        Details:
            x = [[h1, h2, ... hn]]

            Where each h is a vector of conditioning features

            Eg: target=2, overlap=1 with x.size(1)=10

            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        '''

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')

        folded = x.new_zeros(num_folds, target + 2 * overlap, features)

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    def get_step(self):
        return self.step.data.item()

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([cumproduct(p.size())[-1] for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)
        return parameters

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns used by the WaveRNN. Used
        to improve efficiency and avoid PyTorch yelling at us."""
        [m.flatten_parameters() for m in self._to_flatten]
