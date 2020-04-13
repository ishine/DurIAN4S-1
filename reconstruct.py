import sys 
sys.path.append('model')
sys.path.append('extension')

import torch
import torchaudio
from termcolor import colored

import dsp_xt as dsp
from file_xt import FileXT
from config_xt import ConfigXT
from torch_xt import set_device, load_checkpoint
from wavernn import WaveRNN
from glow import WaveGlow, Denoiser

def wavernn_infer(mel, config):
    print(colored('Running WaveRNN with ', 'blue', attrs=['bold']) + config.vocoder_path)
    wavernn = WaveRNN(config)
    wavernn.load_state_dict(torch.load(config.vocoder_path, map_location='cpu'))
    wavernn = set_device(wavernn, config.device)

    wave = wavernn.infer(mel)

    return wave.cpu()

def waveglow_infer(mel, config):
    print(colored('Running WaveGlow with ', 'blue', attrs=['bold']) + config.vocoder_path)
    '''
    waveglow = WaveGlow(config)
    waveglow = load_checkpoint(config.vocoder_path, waveglow)
    '''
    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = set_device(waveglow, config.device)
    waveglow.eval()

    denoiser = Denoiser(waveglow, config)
    denoiser = set_device(denoiser, config.device)

    with torch.no_grad():
        wave = waveglow.infer(mel, config.sigma).float()
        wave = denoiser(wave, strength=config.denoising_strength)

    wave = wave/torch.max(torch.abs(wave))

    return wave.cpu()

def melgan_infer(mel, config):
    print(colored('Running MelGAN with ', 'blue', attrs=['bold']) + config.vocoder_path)
    melgan = torch.hub.load('seungwonpark/melgan', 'melgan')
    melgan = set_device(melgan, config.device)
    melgan.eval()

    with torch.no_grad():
        wave = melgan.inference(mel)

    max_wav_value = 2**(16 - 1) # For 16bit audio
    wave = wave.float()/max_wav_value

    return wave.cpu()

def main():
    config = ConfigXT()
    load = FileXT(config.audio_path)

    print(colored('Preprocessing audio for ', 'blue', attrs=['bold']) + load.basename)
    y = dsp.load(load.filename, config.sample_rate)
    mel = dsp.melspectrogram(y, config, squeeze=False)
    mel = set_device(mel, config.device)

    if config.vocoder == 'wavernn':
        wave = wavernn_infer(mel, config)
    elif config.vocoder == 'waveglow':
        wave = waveglow_infer(mel, config)
    elif config.vocoder == 'melgan':
        wave  = melgan_infer(mel, config)

    savename = config.vocoder_path.replace('.pt', '_') + load.basename
    torchaudio.save(savename, wave, config.sample_rate)

    print(colored('Audio generated to ', 'blue', attrs=['bold']) + savename)

if __name__ == "__main__":
    main()