import sys 
sys.path.append('model')
sys.path.append('extension')

import torch
import torchaudio
from tqdm import tqdm
from termcolor import colored

import preprocess
import dataprocess
import dsp_xt as dsp
from config_xt import ConfigXT
from file_xt import FileXT
from torch_xt import set_device, load_checkpoint
from acoustic import Acoustic
from durian import DurIAN
from wavernn import WaveRNN 
from reconstruct import wavernn_infer, waveglow_infer

def acoustic_infer(data, config):
    dataloader = dataprocess.load_infer(data, model_type='acoustic')
    model = Acoustic(config)
    model.load_state_dict(torch.load(config.acoustic_path, map_location='cpu')['state_dict'])
    model = set_device(model, config.device)
    model.eval() 

    print(colored('Generating acoustics with ', 'blue', attrs=['bold']) + config.acoustic_path)

    f0_min = dsp.midi2hz(config.min_note + 3)
    f0_min = dsp.f0_normalize(f0_min, config.min_note, config.min_note + config.num_note)

    f0 = []
    rmse = []
    y_prev = set_device((torch.zeros(1, 1), torch.zeros(1, 1)), config.device)
    for batch in tqdm(dataloader, leave=False, ascii=True):
        x, y_prev, _ = set_device(batch, config.device)

        f0_gen, rmse_gen = model(x, y_prev)
        y_prev = (f0_gen, rmse_gen)

        f0_denoised = f0_gen.squeeze(0).data
        f0_denoised[f0_denoised < f0_min] = 0
        f0.append(f0_denoised)

        rmse_denoised = rmse_gen.squeeze(0).data
        rmse_denoised[rmse_denoised < 0] = 0
        rmse.append(rmse_denoised)

    return f0, rmse

def durian_infer(data, config):
    dataloader = dataprocess.load_infer(data, model_type='durian')
    model = DurIAN(config)
    model.load_state_dict(torch.load(config.durian_path, map_location='cpu')['state_dict'])
    model = set_device(model, config.device)
    model.eval()

    print(colored('Generating mel-spectrogram with ', 'blue', attrs=['bold']) + config.durian_path)
    mel = []
    y_prev = set_device(torch.zeros(1, config.mel_size, 1), config.device)
    for batch in tqdm(dataloader, leave=False, ascii=True):
        x, _, _ = set_device(batch, config.device)
        
        y_gen, _ = model(x, y_prev)
        mel.append(y_gen.data)
        y_prev = y_gen[...,-1].unsqueeze(-1)

    mel = torch.cat(mel, dim=-1)

    return mel

def main():
    config = ConfigXT()
    load = FileXT(config.audio_path)

    print(colored('Preprocessing audio for ', 'blue', attrs=['bold']) + load.basename)
    data = preprocess.preprocess(load.filename, config.speaker, config, verbose=False)
    if config.infer_acoustic: 
        f0, rmse = acoustic_infer(data, config)
        for i in range(len(data)):
            data[i][4] = f0[i]
            data[i][5] = rmse[i]

    mel = durian_infer(data, config)

    if config.vocoder == 'wavernn':
        wave = wavernn_infer(mel, config)
    elif config.vocoder == 'waveglow':
        wave = waveglow_infer(mel, config)

    savename = config.durian_path.replace('.pt', '_') + FileXT(config.vocoder_path).basestem + '_speaker' + str(config.speaker) + '_' + load.basename
    torchaudio.save(savename, wave, config.sample_rate)

    print(colored('Audio generated to ', 'blue', attrs=['bold']) + savename + '\n')
    
if __name__ == "__main__":
    main()