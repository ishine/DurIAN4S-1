import sys 
sys.path.append('model')
sys.path.append('extension')

import torch
import torchaudio
from termcolor import colored

import dsp_xt as dsp
from file_xt import FileXT
from config_xt import ConfigXT
from torch_xt import set_device
from wavernn import WaveRNN

def main():
    config = ConfigXT()
    load = FileXT(config.audio_path)

    print(colored('Preprocessing audio for ', 'blue', attrs=['bold']) + load.basename)
    y = dsp.load(load.filename, config.sample_rate)
    mel = dsp.melspectrogram(y, config, squeeze=False)
    mel = set_device(mel, config.device)

    print(colored('Running WaveRNN with ', 'blue', attrs=['bold']) + config.wavernn_path)
    model = WaveRNN(config)
    model.load_state_dict(torch.load(config.wavernn_path, map_location='cpu'), strict=False)
    model = set_device(model, config.device)

    y_rec = model.infer(mel).cpu()
    savename = config.wavernn_path.replace('.pt', '_') + load.basename
    torchaudio.save(savename, y_rec, config.sample_rate)

    print(colored('Audio generated to ', 'blue', attrs=['bold']) + savename)

if __name__ == "__main__":
    main()