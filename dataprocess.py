import os
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DataSplit(object):
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test

class SingleLoader(Dataset):
    def __init__(self, path, set_name, config):
        self.data = torch.load(os.path.join(path, set_name + '.pt'))
        self.batch_size = config.batch_size
        self.test_run = config.test_run 

    def __getitem__(self, index):
        phoneme = self.data[index][0]
        speaker = self.data[index][1]
        duration = self.data[index][2]
        f0 = self.data[index][3]
        rmse = self.data[index][4]
        position = self.data[index][5]
        mel_prev = self.data[index][6][:,:-1]
        mel = self.data[index][6][:,1:]

        return phoneme, speaker, duration, f0, rmse, position, mel_prev, mel

    def __len__(self):
        if self.test_run:
            return min(2*self.batch_size, len(self.data))
        else:
            return len(self.data)

class InferLoader(Dataset): 
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        phoneme = self.data[index][0]
        speaker = self.data[index][1]
        duration = self.data[index][2]
        f0 = self.data[index][3]
        rmse = self.data[index][4]
        position = self.data[index][5]
        mel_prev = self.data[index][6][:,:-1]
        mel = self.data[index][6][:,1:]

        x = (phoneme, (speaker, duration, f0, rmse, position))

        return x, mel_prev, mel

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    phoneme, speaker, duration, f0, rmse, position, mel_prev, mel = zip(*batch)

    batch_size = len(batch)
    mel_size = mel[0].size(0)
    p_lens = torch.zeros(len(batch)).long()
    m_lens = torch.zeros(len(batch)).long()

    for i in range(len(batch)):
        p_lens[i] = phoneme[i].size(0)
        m_lens[i] = mel[i].size(1)

    m_lens, indices = torch.sort(m_lens, descending=True)
    p_lens = p_lens[indices]

    phoneme_padded = torch.zeros(batch_size, max(p_lens)).long()
    speaker_padded = torch.zeros(batch_size, max(p_lens)).long()
    duration_padded = torch.zeros(batch_size, max(p_lens)).long()
    f0_padded = torch.zeros(batch_size, max(m_lens))
    rmse_padded = torch.zeros(batch_size, max(m_lens))
    position_padded = torch.zeros(batch_size, max(m_lens))

    mel_prev_padded = torch.zeros(batch_size, mel_size, max(m_lens))
    mel_padded = torch.zeros(batch_size, mel_size, max(m_lens))

    for i in range(batch_size):
        p_len = p_lens[i].item()
        m_len = m_lens[i].item()
        phoneme_padded[i,:p_len] = phoneme[indices[i]]
        speaker_padded[i,:p_len] = speaker[indices[i]] 
        duration_padded[i,:p_len] = duration[indices[i]]
        f0_padded[i,:m_len] = f0[indices[i]]
        rmse_padded[i,:m_len] = rmse[indices[i]]
        position_padded[i,:m_len] = position[indices[i]]

        mel_prev_padded[i,:,:m_len] = mel_prev[indices[i]]
        mel_padded[i,:,:m_len] = mel[indices[i]]

    x = (phoneme_padded, (speaker_padded, duration_padded, f0_padded, rmse_padded, position_padded))

    return x, mel_prev_padded, mel_padded

def load_train(config):
    dataset_train = SingleLoader(config.feature_path, 'train', config)
    dataset_valid = SingleLoader(config.feature_path, 'valid', config)

    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                                  collate_fn=collate_fn, drop_last=False, num_workers=config.num_proc)
    dataloader_valid = DataLoader(dataset_valid, batch_size=config.batch_size, shuffle=False,
                                  collate_fn=collate_fn, drop_last=False, num_workers=config.num_proc)

    return DataSplit(dataloader_train, dataloader_valid, None)

def load_infer(data):
    dataset = InferLoader(data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    return dataloader 