import os
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DataSplit(object):
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test

class SingleLoaderDurIAN(Dataset):
    def __init__(self, path, set_name, config):
        self.data = torch.load(os.path.join(path, set_name + '.pt'))
        self.batch_size = config.batch_size
        self.test_run = config.test_run 

    def __getitem__(self, index):
        phoneme = self.data[index][1]
        speaker = self.data[index][2]
        duration = self.data[index][3]
        f0 = self.data[index][4][1:]
        rmse = self.data[index][5][1:]
        position = self.data[index][6]
        mel_prev = self.data[index][7][:,:-1]
        mel = self.data[index][7][:,1:]

        return phoneme, speaker, duration, f0, rmse, position, mel_prev, mel

    def __len__(self):
        if self.test_run:
            return min(2*self.batch_size, len(self.data))
        else:
            return len(self.data)

class SingleLoaderAcoustic(Dataset):
    def __init__(self, path, set_name, config):
        self.data = torch.load(os.path.join(path, set_name + '.pt'))
        self.batch_size = config.batch_size
        self.test_run = config.test_run 

    def __getitem__(self, index):
        note = self.data[index][0]
        speaker = self.data[index][2]
        f0_prev = self.data[index][4][:-1]
        f0 = self.data[index][4][1:]
        rmse_prev = self.data[index][5][:-1]
        rmse = self.data[index][5][1:]

        return note, speaker, f0_prev, f0, rmse_prev, rmse

    def __len__(self):
        if self.test_run:
            return min(2*self.batch_size, len(self.data))
        else:
            return len(self.data)

class InferLoaderDurIAN(Dataset): 
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        phoneme = self.data[index][1]
        speaker = self.data[index][2]
        duration = self.data[index][3]
        f0 = self.data[index][4][1:]
        rmse = self.data[index][5][1:]
        position = self.data[index][6]
        mel_prev = self.data[index][7][:,:-1]
        mel = self.data[index][7][:,1:]

        x = (phoneme, (speaker, duration, f0, rmse, position))

        return x, mel_prev, mel

    def __len__(self):
        return len(self.data)

class InferLoaderAcoustic(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        note = self.data[index][0]
        speaker = self.data[index][2]
        f0_prev = self.data[index][4][:-1]
        f0 = self.data[index][4][1:]
        rmse_prev = self.data[index][5][:-1]
        rmse = self.data[index][5][1:]

        x = (note, speaker)
        y_prev = (f0_prev, rmse_prev)
        y = (f0, rmse)

        return x, y_prev, y

    def __len__(self):
        return len(self.data)

def collate_durian(batch):
    phoneme, speaker, duration, f0, rmse, position, mel_prev, mel = zip(*batch)

    batch_size = len(batch)
    mel_size = mel[0].size(0)
    x_lens = torch.zeros(len(batch)).long()
    y_lens = torch.zeros(len(batch)).long()

    for i in range(len(batch)):
        x_lens[i] = phoneme[i].size(0)
        y_lens[i] = mel[i].size(1)

    y_lens, indices = torch.sort(y_lens, descending=True)
    x_lens = x_lens[indices]

    phoneme_padded = torch.zeros(batch_size, max(x_lens)).long()
    speaker_padded = torch.zeros(batch_size, max(x_lens)).long()
    duration_padded = torch.zeros(batch_size, max(x_lens)).long()
    f0_padded = torch.zeros(batch_size, max(y_lens))
    rmse_padded = torch.zeros(batch_size, max(y_lens))
    position_padded = torch.zeros(batch_size, max(y_lens))

    mel_prev_padded = torch.zeros(batch_size, mel_size, max(y_lens))
    mel_padded = torch.zeros(batch_size, mel_size, max(y_lens))

    for i in range(batch_size):
        x_len = x_lens[i].item()
        y_len = y_lens[i].item()
        phoneme_padded[i,:x_len] = phoneme[indices[i]]
        speaker_padded[i,:x_len] = speaker[indices[i]] 
        duration_padded[i,:x_len] = duration[indices[i]]
        f0_padded[i,:y_len] = f0[indices[i]]
        rmse_padded[i,:y_len] = rmse[indices[i]]
        position_padded[i,:y_len] = position[indices[i]]

        mel_prev_padded[i,:,:y_len] = mel_prev[indices[i]]
        mel_padded[i,:,:y_len] = mel[indices[i]]

    x = (phoneme_padded, (speaker_padded, duration_padded, f0_padded, rmse_padded, position_padded))

    return x, mel_prev_padded, mel_padded

def collate_acoustic(batch):
    note, speaker, f0_prev, f0, rmse_prev, rmse = zip(*batch)

    batch_size = len(batch)
    x_lens = torch.zeros(len(batch)).long()
    y_lens = torch.zeros(len(batch)).long()

    for i in range(len(batch)):
        x_lens[i] = note[i].size(0)
        y_lens[i] = f0[i].size(0)

    y_lens, indices = torch.sort(y_lens, descending=True)
    x_lens = x_lens[indices]

    note_padded = torch.zeros(batch_size, max(x_lens)).long()
    f0_prev_padded = torch.zeros(batch_size, max(y_lens))
    f0_padded = torch.zeros(batch_size, max(y_lens))
    rmse_prev_padded = torch.zeros(batch_size, max(y_lens))
    rmse_padded = torch.zeros(batch_size, max(y_lens))

    for i in range(batch_size):
        x_len = x_lens[i].item()
        y_len = y_lens[i].item()

        note_padded[i,:x_len] = note[indices[i]]
        f0_prev_padded[i,:y_len] = f0_prev[indices[i]]
        f0_padded[i,:y_len] = f0[indices[i]]
        rmse_prev_padded[i,:y_len] = rmse_prev[indices[i]]
        rmse_padded[i,:y_len] = rmse[indices[i]]

    x = (note_padded, speaker)
    y_prev = (f0_prev_padded, rmse_prev_padded)
    y = (f0_padded, rmse_padded)
    
    return x, y_prev, y

def load_train(config, model_type='durian'):
    if model_type is 'durian':
        TrainLoader = SingleLoaderDurIAN
        collate_fn = collate_durian
    elif model_type is 'acoustic':
        TrainLoader = SingleLoaderAcoustic
        collate_fn = collate_acoustic
    else: 
        raise AssertionError('Please use valid model type.')

    dataset_train = TrainLoader(config.feature_path, 'train', config)
    dataset_valid = TrainLoader(config.feature_path, 'valid', config)

    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                                  collate_fn=collate_fn, drop_last=False, num_workers=config.num_proc)
    dataloader_valid = DataLoader(dataset_valid, batch_size=config.batch_size, shuffle=False,
                                  collate_fn=collate_fn, drop_last=False, num_workers=config.num_proc)

    return DataSplit(dataloader_train, dataloader_valid, None)

def load_infer(data, model_type='durian'):
    if model_type is 'durian':
        InferLoader = InferLoaderDurIAN
    elif model_type is 'acoustic':
        InferLoader = InferLoaderAcoustic
    else: 
        raise AssertionError('Please use valid model type.')

    dataset = InferLoader(data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    return dataloader 