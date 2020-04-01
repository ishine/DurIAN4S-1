import sys 
sys.path.append('model')
sys.path.append('extension')

import torch
import torchaudio

import dsp_xt as dsp
from file_xt import FileXT
from config_xt import ConfigXT
from torch_xt import set_device
from wavernn import WaveRNN

def main():
    config = ConfigXT()

    load = FileXT(config.audio_path)
    y, sample_rate = torchaudio.load(load.filename)
    mel = dsp.melspectrogram(y, config, squeeze=False)
    mel = set_device(mel, config.device)

    model = WaveRNN(config)
    model.load_state_dict(torch.load(config.wavernn_path, map_location='cpu'), strict=False)
    model = set_device(model, config.device)

    y_rec = model.generate(mel, config.batched, config.target_samples, config.overlap, config.mu_law).cpu()
    save = FileXT(config.model_path.replace('.pt', '_') + load.basename)
    torchaudio.save(save.filename, y_rec, config.sample_rate)

    print(save.filename)

if __name__ == "__main__":
    main()