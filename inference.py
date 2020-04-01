import sys 
sys.path.append('model')
sys.path.append('extension')

import torch
import torchaudio
from tqdm import tqdm

import preprocess
import dataprocess
from config_xt import ConfigXT
from file_xt import FileXT
from torch_xt import set_device, load_checkpoint
from tacotron import Tacotron
from wavernn import WaveRNN 

def main():
    config = ConfigXT()

    wav = FileXT(config.audio_path)
    data = preprocess.preprocess(wav.filename, config.speaker, config)
    dataloader = dataprocess.load_infer(data)

    model = Tacotron(config)
    model.load_state_dict(torch.load(config.model_path, map_location='cpu'), strict=False)
    model = set_device(model, config.device)
    model.eval()

    criterion = torch.nn.L1Loss()

    mel = []
    #y_prev = set_device(torch.zeros(1, config.mel_size, 1), config.device)
    for batch in tqdm(dataloader, leave=False, ascii=True):
        x, y_prev, y = set_device(batch, config.device)
        y_gen, y_decoder_gen = model(x, y_prev)

        loss = criterion(y_gen, y) + criterion(y_decoder_gen, y)
        y_gen = y_gen.clamp(0, 1)

        mel.append(y_gen.data.cpu())
        #y_prev = y_gen[...,-1].unsqueeze(-1)

    mel = torch.cat(mel, dim=-1)
    wavernn = WaveRNN(config)
    wavernn.load_state_dict(torch.load(config.wavernn_path, map_location='cpu'), strict=False)

    wave = wavernn.generate(mel, config.batched, config.target_samples, config.overlap, config.mu_law).cpu()
    save = FileXT(config.model_path.replace('.pt', '_') + wav.basename)
    torchaudio.save(save.filename, wave, config.sample_rate)

    print('Audio generated to \'%s\'' % (save.filename))
    
if __name__ == "__main__":
    main()