import torch
import torch.nn as nn

from layers import Prenet, CBHG

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.num_char, config.text_embed_size)
        self.prenet = Prenet(config.text_embed_size, config.prenet_size)
        self.cbhg = CBHG(input_size=config.encoder_size,
                         projections=[config.encoder_size, config.encoder_size],
                         K=config.encoder_K,
                         num_highways=config.num_highways)

    def forward(self, x):
        x = self.embedding(x)
        x = self.prenet(x)
        x = self.cbhg(x)

        return x 

class Alignment(nn.Module):
    def __init__(self, config):
        super(Alignment, self).__init__()
        self.speaker_embedding = nn.Embedding(config.num_speaker, config.speaker_embed_size)
        self.linear = nn.Linear(2*config.encoder_size + config.speaker_embed_size, config.alignment_size)

    def forward(self, x, conditions, max_y_len):
        speaker, duration, f0, rmse, position = conditions

        batch_size = x.size(0)
        speaker = self.speaker_embedding(speaker)
        x = torch.cat((x, speaker), dim=-1)
        x = self.linear(x)

        x_expanded = x.new_zeros((batch_size, max_y_len, x.size(2)))
        for i in range(batch_size):
            y_len = torch.sum(duration[i])
            x_expanded[i,:y_len] = x[i].repeat_interleave(duration[i], dim=0)

        f0 = f0.unsqueeze(-1)
        rmse = rmse.unsqueeze(-1)
        position = position.unsqueeze(-1)

        return torch.cat((x_expanded, f0, rmse, position), dim=-1)

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.mel_size = config.mel_size
        self.decoder_size = config.decoder_size
        self.reduction = config.reduction

        self.prenet = Prenet(config.mel_size, config.prenet_size)
        self.rnn_projection = nn.Linear(config.alignment_size + 3 + config.prenet_size[-1], config.decoder_size)
        self.decoder_rnn = nn.ModuleList(
            [nn.GRUCell(config.decoder_size, config.decoder_size) for _ in range(2)])
        self.mel_projection = nn.Linear(config.decoder_size, config.reduction*config.mel_size)

    def initialize(self, x):
        batch_size = x.size(0)
        self.decoder_hidden = len(self.decoder_rnn)*[x.new_zeros(batch_size, self.decoder_size)]

    def decode(self, x):
        for i in range(len(self.decoder_rnn)):
            self.decoder_hidden[i] = self.decoder_rnn[i](x, self.decoder_hidden[i])
            x = x + self.decoder_hidden[i]

        return x
    
    def forward(self, x, mel_inputs):
        self.initialize(x)
        
        x = x.transpose(0, 1)
        mel_inputs = mel_inputs.transpose(0, 1)
        mel_input = mel_inputs[0]
        
        mel_outputs = []
        for i in range(x.size(0)):
            mel_input = self.prenet(mel_input)
            decoder_input = torch.cat((mel_input, x[i]), dim=-1)
            decoder_input = self.rnn_projection(decoder_input)

            mel_output = self.decode(decoder_input)
            mel_output = self.mel_projection(mel_output)
            mel_output = mel_output.view(self.reduction, -1, self.mel_size)
            mel_outputs.append(mel_output)

            # Teacher forcing for training 
            if mel_inputs.size(0) == 1:
                mel_input = mel_output[-1]
            elif mel_inputs.size(0) > 1 and i < mel_inputs.size(0) - 1:
                mel_input = mel_inputs[i+1]

        mel_outputs = torch.cat(mel_outputs).transpose(0, 1)

        return mel_outputs

class Tacotron(nn.Module):
    def __init__(self, config):
        super(Tacotron, self).__init__()
        self.encoder = Encoder(config)
        self.alignment_layer = Alignment(config)
        self.decoder = Decoder(config)
        self.postnet = CBHG(input_size=config.mel_size,
                            projections=[config.postnet_size, config.mel_size],
                            K=config.postnet_K,
                            num_highways=config.num_highways)
        self.post_projection = nn.Linear(2*config.mel_size, config.mel_size)

    def forward(self, x, y_prev):
        phoneme, conditions = x
        if y_prev.size(-1) > 1: # When ground truth is given
            max_y_len = y_prev.size(-1)
        else: # When length of f0 is given for inference
            max_y_len = conditions[2].size(-1)

        encoder_output = self.encoder(phoneme)
        aligned_output = self.alignment_layer(encoder_output, conditions, max_y_len)
        
        y_prev = y_prev.transpose(1, 2)
        y_decoder = self.decoder(aligned_output, y_prev)
        y = self.postnet(y_decoder)
        y = self.post_projection(y)

        return y.transpose(1, 2), y_decoder.transpose(1, 2)
