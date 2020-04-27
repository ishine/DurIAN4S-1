import sys 
sys.path.append('model')
sys.path.append('extension')

import torch
import torchaudio
from tqdm import tqdm
from termcolor import colored

import preprocess
import dataprocess
from config_xt import ConfigXT
from file_xt import FileXT
from torch_xt import set_device, load_checkpoint
from tacotron import Tacotron
from wavernn import WaveRNN 
from reconstruct import wavernn_infer, waveglow_infer

def main():
    config = ConfigXT()
    load = FileXT(config.audio_path)

    print(colored('Preprocessing audio for ', 'blue', attrs=['bold']) + load.basename)
    data = preprocess.preprocess(load.filename, config.speaker, config, verbose=False)
    dataloader = dataprocess.load_infer(data)

    model = Tacotron(config)
    model.load_state_dict(torch.load(config.model_path, map_location='cpu')['state_dict'])
    model = set_device(model, config.device)
    model.eval()

    print(colored('Generating mel-spectrogram with ', 'blue', attrs=['bold']) + config.model_path)
    mel = []
    y_prev = set_device(torch.zeros(1, config.mel_size, 1), config.device)
    for batch in tqdm(dataloader, leave=False, ascii=True):
        x, y_prev, _ = set_device(batch, config.device)
        
        y_gen, _ = model(x, y_prev)
        mel.append(y_gen.data)
        y_prev = y_gen[...,-1].unsqueeze(-1)

    mel = torch.cat(mel, dim=-1)
    if config.vocoder == 'wavernn':
        wave = wavernn_infer(mel, config)
    elif config.vocoder == 'waveglow':
        wave = waveglow_infer(mel, config)

    savename = config.model_path.replace('.pt', '_') + FileXT(config.vocoder_path).basestem + '_speaker' + str(config.speaker) + '_' + load.basename
    torchaudio.save(savename, wave, config.sample_rate)

    print(colored('Audio generated to ', 'blue', attrs=['bold']) + savename + '\n')
    
if __name__ == "__main__":
    main()