import torch
import torch.nn.functional as F
import torchaudio
import pyworld as pw 
from torchaudio.transforms import Resample, Spectrogram
from librosa import filters

from torch_xt import set_device

# Scale Methods
def time2frame(x, frame_rate):
    return int(x*frame_rate)

def frame(x, win_size, hop_size):
    if x.dim() == 1:
        num_frame = (x.size(0) - win_size)//hop_size + 1
        y = x.new_zeros(win_size, num_frame)
        for i in range(num_frame):
            y[:,i] = x[hop_size*i:hop_size*i + win_size]
    elif x.dim() == 2:
        num_frame = (x.size(1) - win_size)//hop_size + 1
        y = x.new_zeros(x.size(0), win_size, num_frame)
        for i in range(x.size(0)):
            for j in range(num_frame):
                y[i,:,j] = x[i, hop_size*j:hop_size*j + win_size]
    else:
        raise AssertionError("Input dimension should be 1 or 2")

    return y

def to_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float)

    return x 

def amp2db(x, min_level_db=None):
    x = to_tensor(x)

    x = 20.0*torch.log10(x)
    if min_level_db is not None:
        x = torch.max(x, min_level_db)

    return x 

def db2amp(x):
    x = to_tensor(x)

    return torch.pow(10.0, x*0.05)

def hz2midi(x):
    x = to_tensor(x)

    return 69.0 + 12.0*torch.log2(x/440.0)

def midi2hz(x):
    x = to_tensor(x)

    return 440.0*torch.pow(2.0, (x - 69)/12)

def normalize(x, min_level_db):
    return torch.clamp((x - min_level_db)/(-min_level_db), 0, 1)

def denormalize(x, min_level_db):
    return x.clamp(0, 1)*(-min_level_db) + min_level_db

def f0_normalize(x, min_note, max_note):
    x = to_tensor(x)

    x = hz2midi(x)
    x = (x - min_note)/(max_note - min_note)

    return torch.clamp(x, 0, 1)

def f0_denormalize(x, min_note, max_note):
    x = to_tensor(x)

    x = x.clamp(0, 1)
    x = (max_note - min_note)*x + min_note

    return midi2hz(x)

# F0 Methods 
def world(y, sample_rate, fft_size, hop_size):
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    if y.ndim == 2:
        y = y.squeeze(0)

    y = y.astype('float64')
    frame_period = 1000*hop_size/sample_rate
    f0, sp, ap = pw.wav2world(y, sample_rate, fft_size=fft_size, frame_period=frame_period)
    f0 = torch.from_numpy(f0).float()
    sp = torch.from_numpy(sp).float()
    ap = torch.from_numpy(ap).float()

    return f0, sp, ap

# Wave Methods
def load(filename, sample_rate):
    y, source_rate = torchaudio.load(filename)
    if source_rate != sample_rate:
        resample = Resample(source_rate, sample_rate)
        y = resample(y)

    return y 

def rmse(y, win_size, hop_size, center=True, pad_mode='reflect', squeeze=True):
    if center:
        y = y.unsqueeze(0)
        y = F.pad(y, (win_size//2, win_size//2), pad_mode, 0)
        y = y.squeeze(0)

    y = frame(y, win_size, hop_size)
    rmse = torch.sqrt(torch.mean(torch.abs(y)**2, dim=y.dim() - 2))
    if squeeze:
        rmse = rmse.squeeze(0)

    return rmse

# Spectral Methods
def stft(y, config): 
    spec_fn = Spectrogram(n_fft=config.fft_size, 
                          win_length=config.win_size, 
                          hop_length=config.hop_size)
    y, spec_fn = set_device((y, spec_fn), config.device)
    spec = torch.sqrt(spec_fn(y))

    return spec

def istft(magnitude, phase, config):
    window = torch.hann_window(config.win_size)
    stft_matrix = torch.stack((magnitude*torch.cos(phase), magnitude*torch.sin(phase)), dim=-1)
    stft_matrix, window = set_device((stft_matrix, window), config.device)
    y = torchaudio.functional.istft(stft_matrix,
                                    n_fft=config.fft_size,
                                    hop_length=config.hop_size,
                                    win_length=config.win_size,
                                    window=window)

    return y

def magphase(y, config):
    window = torch.hann_window(config.win_size)
    y, window = set_device((y, window), config.device)
    stft_matrix = torch.stft(y, 
                             n_fft=config.fft_size, 
                             hop_length=config.hop_size, 
                             win_length=config.fft_size,
                             window=window)
    
    real = stft_matrix[...,0]
    imag = stft_matrix[...,1]

    magnitude = torch.sqrt(real**2 + imag**2)
    phase = torch.atan2(imag.data, real.data)

    return magnitude, phase

def spectrogram(y, config, squeeze=True):
    spec = stft(y, config)
    if config.norm_type == 'db':
        spec = amp2db(spec) - config.ref_level_db
        spec = normalize(spec, config.min_level_db)
    elif config.norm_type == 'log':
        min_level = db2amp(config.min_level_db)
        spec = torch.log(torch.clamp(spec, min=min_level))
    else:
        raise AssertionError("Invalid normalization type!")

    if squeeze:
        spec = spec.squeeze(0)
    
    return spec

def melspectrogram(y, config, squeeze=True):
    spec = stft(y, config)
    mel_filter = filters.mel(sr=config.sample_rate,
                             n_fft=config.fft_size,
                             n_mels=config.mel_size,
                             fmin=config.mel_fmin,
                             fmax=config.mel_fmax)
    mel_filter = torch.from_numpy(mel_filter)
    mel_filter = set_device(mel_filter, config.device)
    mel = torch.matmul(mel_filter, spec)
    if config.norm_type == 'db':
        mel = normalize(amp2db(mel), config.min_level_db)
    elif config.norm_type == 'log':
        min_level = db2amp(config.min_level_db)
        mel = torch.log(torch.clamp(mel, min=min_level))
        
    if squeeze:
        mel = mel.squeeze(0)

    return mel
