import sys
sys.path.append('extension')
sys.path.append('g2p')

import os 
import torch 
import torchaudio
from madmom.io.midi import load_midi
from multiprocessing import Pool
from functools import partial

import korean_g2p
import dsp_xt as dsp 
from config_xt import ConfigXT
from file_xt import FileXT, create_path

class Range(object):
    def __init__(self, start=None, duration=None, frame_rate=1):
        if start is None:
            self.start = 0
        else:
            self.start = dsp.time2frame(start, frame_rate)

        if duration is None:
            self.end = start
            self.duration = 0
        else:
            self.end = dsp.time2frame(start + duration, frame_rate)
            self.duration = self.end - self.start
    
    def to_arange(self):
        return torch.arange(self.start, self.end)

class SegmentList(object):
    def __init__(self):
        self.list = []
        self.segment = []

    def extend(self, x):
        self.segment.extend(x)

    def split(self):
        tensor = torch.FloatTensor
        if type(self.segment[0]) is int:
            tensor = torch.LongTensor 
        
        self.segment = tensor(self.segment)
        self.list.append(self.segment)
        self.segment = []

def load_wav(filename, config):
    y = dsp.load(filename, config.sample_rate)

    f0, _, _ = dsp.world(y, config.sample_rate, config.fft_size, config.hop_size)
    f0 = dsp.f0_normalize(f0, config.min_note, config.min_note + config.num_note)
    rmse = dsp.rmse(y, config.win_size, config.hop_size)

    mel = dsp.melspectrogram(y, config).cpu()
    mel = torch.cat((torch.zeros(config.mel_size, 1), mel), dim=1) # For previous frame

    return f0, rmse, mel

'''
def load_mid(filename):
    midi_file = MIDIFile.from_file(filename)
    midi = midi_file.notes(unit='seconds')

    return midi
'''

def load_text(filename):
    text_file = open(filename)
    text = text_file.read().replace(' ', '').replace('\n', '')
    text = korean_g2p.encode(text)

    return text

def get_duration(phone, note_duration, length_c):
    duration = []
    if note_duration < phone.num():
        length_c = 0
    elif note_duration <= phone.num()*length_c:
        length_c = max(note_duration//phone.num() - 1, 1)
    
    if phone.ons is not None:
        duration.append(length_c)
    if phone.nuc is not None:
        length_v = note_duration - (phone.num() - 1)*length_c
        duration.append(length_v)
    if phone.cod is not None:
        duration.append(length_c)

    return duration

def get_position(note_duration):
    position = []
    for i in range(note_duration):
        position.append(i/note_duration)

    return position

def segment_validation(duration, mel_range):
    total_length = 0
    for d in duration.segment:
        total_length += d 
    
    invalid = False
    if total_length != (mel_range.end - mel_range.start):
        invalid = True

    return invalid

def get_alignment(text, midi, config):
    frame_rate = config.sample_rate/config.hop_size
    phoneme = SegmentList()
    duration = SegmentList()
    position = SegmentList()
    mel_range_list = []

    mel_range = Range()
    for i in range(len(midi)):
        # Remove MIDI overlap
        if i < len(midi) - 1:
            if midi[i][0] + midi[i][2] > midi[i+1][0]:
                midi[i][2] = midi[i+1][0] - midi[i][0]

        prev_range = Range(0, 0, frame_rate)
        curr_range = Range(midi[i][0], midi[i][2], frame_rate)
        next_range = Range(midi[-1][0] + midi[-1][2], None, frame_rate)
        if i > 0:
            prev_range = Range(midi[i-1][0], midi[i-1][2], frame_rate)
        if i < len(midi) - 1:
            next_range = Range(midi[i+1][0], midi[i+1][2], frame_rate)
        
        if len(phoneme.segment) == 0:
            if curr_range.start - mel_range.start > config.max_unvoice:
                mel_range.start = curr_range.start - config.max_unvoice
            prev_range = Range(mel_range.start, 0, 1)

        split = False
        # Condition for stacked mel length
        if next_range.end - mel_range.start < config.min_length: 
            split = False
        elif next_range.end - mel_range.start > config.max_length:
            split = True
        # Condition for interval between midi 
        else:
            if next_range.start - curr_range.end < config.min_unvoice:
                split = False
            else:
                split = True

        if i == len(midi) - 1:
            split = True

        t = text[i].to_list()
        d = get_duration(text[i], curr_range.duration, config.length_c)
        p = get_position(curr_range.duration)

        # Condition for breath attack 
        d_unvoice = curr_range.start - prev_range.end
        if d_unvoice > 0:
            t = [0] + t
            d = [d_unvoice] + d
            p = get_position(d_unvoice) + p 

        phoneme.extend(t)
        duration.extend(d)
        position.extend(p)

        if split:
            mel_range.end = curr_range.end 
            if next_range.start - curr_range.end > config.length_offset:
                phoneme.extend([0])
                duration.extend([config.length_offset])
                position.extend(get_position(config.length_offset))
                mel_range.end = curr_range.end + config.length_offset

            if segment_validation(duration, mel_range):
                raise AssertionError("Invalid segment found!")

            phoneme.split()
            duration.split()
            position.split()

            mel_range_list.append(mel_range.to_arange())
            mel_range.start = curr_range.end
            if next_range.start - curr_range.end > config.length_offset:
                mel_range.start = curr_range.end + config.length_offset

    return phoneme.list, duration.list, position.list, mel_range_list

def find_error(inputs):
    phoneme, speaker, duration, f0, rmse, position, mel = inputs 
    error = False
    template = "Size mismatch in index"
    error_message = []
    for i in range(len(phoneme)):
        p_len = phoneme[i].size(0)
        m_len = mel[i].size(1) - 1
        if speaker[i].size(0) != p_len:
            error_message.append("%s %d! speaker size: %d, phoneme size: %d" % (template, i, speaker[i].size(0), p_len))
            error = True
        if duration[i].size(0) != p_len:
            error_message.append("%s %d! duration size: %d, phoneme size: %d" % (template, i, duration[i].size(0), p_len))
            error = True
        if f0[i].size(0) != m_len:
            error_message.append("%s %d! f0 size: %d, mel size: %d" % (template, i, f0[i].size(0), m_len))
            error = True
        if rmse[i].size(0) != m_len:
            error_message.append("%s %d! rmse size: %d, mel size: %d" % (template, i, rmse[i].size(0), m_len))
            error = True
        if position[i].size(0) != m_len:
            error_message.append("%s %d! position size: %d, mel size: %d" % (template, i, position[i].size(0), m_len))
            error = True
        
    return error, error_message

def preprocess(filename, index, config, verbose=True):
    basename = filename.split('/')[-1]
    dataset_path = '/'.join(filename.split('/')[:-2])
    txt = FileXT(dataset_path, 'txt', basename, '.txt')
    mid = FileXT(dataset_path, 'mid', basename, '.mid')
    wav = FileXT(dataset_path, 'wav', basename, '.wav')

    text = load_text(txt.filename)
    midi = load_midi(mid.filename)
    features = load_wav(wav.filename, config)
    phoneme, duration, position, mel_range = get_alignment(text, midi, config)

    speaker = []
    f0 = []
    rmse = []
    mel = []
    for i in range(len(mel_range)):
        if mel_range[i][-1].item() >= features[0].size(0):
            mel_range[i] = mel_range[i][mel_range[i] < features[0].size(0)]
            position[i] = position[i][:mel_range[i].size(0)]

        speaker.append(torch.full([phoneme[i].size(0)], fill_value=index).long())
        f0.append(features[0][mel_range[i]])
        rmse.append(features[1][mel_range[i]])

        extended_range = torch.cat((mel_range[i], mel_range[i][-1:] + 1))
        mel.append(features[2][:,extended_range])

    data = phoneme, speaker, duration, f0, rmse, position, mel
    
    if verbose:
        print(basename)

    return transpose(data)

def read_file_list(filename):
    with open(filename) as f:
        file_list = list(f)

    return file_list

def transpose(x):
    return list(map(list, zip(*x)))

def flatten(x):
    flatlist = []
    for sublist in x:
        for item in sublist:
            flatlist.append(item)

    return flatlist

'''
def flatlist(x):
    flatlist = []
    for i in range(len(x[0])):
        var = x[0][i]
        for j in range(1, len(x)):
            var.extend(x[j][i])
        
        flatlist.append(var)
    
    return flatlist
'''
def main():
    config = ConfigXT()
    create_path(config.feature_path, action="override")

    set_list = ['train', 'valid']

    train_var_list = []
    valid_var_list = []
    for i in range(len(config.dataset_path)):
        train_list = read_file_list(os.path.join(config.dataset_path[i], "train_list.txt"))
        valid_list = read_file_list(os.path.join(config.dataset_path[i], "valid_list.txt"))

        if config.num_proc > 1:
            p = Pool(config.num_proc)
            train_var = p.map(partial(preprocess, index=i, config=config), train_list)
            valid_var = p.map(partial(preprocess, index=i, config=config), valid_list)
        else:
            train_var = [preprocess(f, index=i, config=config) for f in train_list]
            valid_var = [preprocess(f, index=i, config=config) for f in valid_list]
        
        train_var_list.append(flatten(train_var))
        valid_var_list.append(flatten(valid_var))

    train_var_list = flatten(train_var_list)
    valid_var_list = flatten(valid_var_list)
    
    dataset = [train_var_list, valid_var_list]
    for i in range(len(set_list)):
        savename = os.path.join(config.feature_path, set_list[i] + '.pt')
        torch.save(dataset[i], savename)

    print("Feature saved to %s" % (config.feature_path))

if __name__ == "__main__":
    main()