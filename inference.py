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
    model.load_state_dict(torch.load(config.model_path, map_location='cpu')['state_dict'])
    model = set_device(model, config.device)
    model.eval()

    mel = []
    y_prev = set_device(torch.zeros(1, config.mel_size, 1), config.device)
    for batch in tqdm(dataloader, leave=False, ascii=True):
        x, y_prev, _ = set_device(batch, config.device)
        
        y_gen, _ = model(x, y_prev)
        mel.append(y_gen.data.cpu())
        y_prev = y_gen[...,-1].unsqueeze(-1)

    mel = torch.cat(mel, dim=-1)
    wavernn = WaveRNN(config)
    wavernn.load_state_dict(torch.load(config.wavernn_path, map_location='cpu'))

    wave = wavernn.generate(mel, config.batched, config.target_samples, config.overlap, config.mu_law).cpu()
    save = FileXT(config.model_path.replace('.pt', '_speaker') + str(config.speaker) + '_' + wav.basename)
    torchaudio.save(save.filename, wave, config.sample_rate)

    print('Audio generated to \'%s\'' % (save.filename))
    
if __name__ == "__main__":
    main()