import torch 
import torch.nn as nn

class Acoustic(nn.Module):
    def __init__(self, config):
        super(Acoustic, self).__init__()
        self.rnn_size = config.aco_rnn_size
        
        self.note_embedding = nn.Embedding(config.num_note, config.note_embed_size)
        self.f0_embedding = nn.Linear(1, config.f0_embed_size)
        self.rmse_embedding = nn.Linear(1, config.rmse_embed_size)

        self.prenet = nn.Sequential(
            nn.Linear(config.note_embed_size, config.aco_prenet_size),
            nn.Tanh(),
            nn.Linear(config.aco_prenet_size, config.aco_prenet_size),
            nn.Tanh())

        self.encoder = nn.GRU(config.aco_prenet_size, config.aco_rnn_size[0], bidirectional=True)

        self.dropout = nn.Dropout(config.aco_dropout)
        self.decoder_f0 = nn.GRUCell(2*config.aco_rnn_size[0] + config.f0_embed_size, config.aco_rnn_size[1])
        self.decoder_rmse = nn.GRUCell(2*config.aco_rnn_size[0] + config.rmse_embed_size, config.aco_rnn_size[1])

        self.postnet_f0 = nn.Linear(config.aco_rnn_size[1], 1)
        self.postnet_rmse = nn.Linear(config.aco_rnn_size[1], 1)

    def initialize(self, x):
        batch_size = x.size(0)
        self.decoder_hidden_f0 = x.new_zeros(batch_size, self.rnn_size[1])
        self.decoder_hidden_rmse = x.new_zeros(batch_size, self.rnn_size[1])

    def encode(self, x, lengths=None):
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        x, _ = self.encoder(x)

        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x 

    def decode(self, x, y_prevs, feature):
        x = x.transpose(0, 1)
        y_prevs = y_prevs.transpose(0, 1)
        y_prev = y_prevs[0]

        y_gens = []
        for i in range(x.size(0)):
            y_prev = self.dropout(y_prev)
            decoder_input = torch.cat((x[i], y_prev), dim=-1)
            if feature is 'f0':
                self.decoder_hidden_f0 = self.decoder_f0(decoder_input, self.decoder_hidden_f0)
                y_gen = self.decoder_hidden_f0.unsqueeze(0)
            elif feature is 'rmse':
                self.decoder_hidden_rmse = self.decoder_rmse(decoder_input, self.decoder_hidden_rmse)
                y_gen = self.decoder_hidden_rmse.unsqueeze(0)
            else:
                raise AssertionError("Please use valid feature type.")
            y_gens.append(y_gen)

            if y_prevs.size(0) == 1:
                y_prev = y_gen 
            elif y_prevs.size(0) > 1 and i < y_prevs.size(0) - 1:
                y_prev = y_prevs[i+1]

        y_gens = torch.cat(y_gens).transpose(0, 1)

        return y_gens

    def forward(self, x, y_prev):
        note, speaker = x 
        f0_prev, rmse_prev = y_prev

        note = self.note_embedding(note)
        f0_prev = self.f0_embedding(f0_prev.unsqueeze(-1))
        rmse_prev = self.rmse_embedding(rmse_prev.unsqueeze(-1))

        self.initialize(note)
        h = self.prenet(note)
        h = self.encode(h)

        f0 = self.decode(h, f0_prev, 'f0')
        f0 = self.postnet_f0(f0).squeeze(-1)

        rmse = self.decode(h, rmse_prev, 'rmse')
        rmse = self.postnet_rmse(rmse).squeeze(-1)

        return f0, rmse
