import json
import torch
from torch.utils.data import Dataset
from dictionary_roll import onseteventstoword, wordtoonsetevents, wordtoint, inttoword, pad_token
import random
import os

class PianoRollDataset(Dataset):
    def __init__(self, folder_path, seq_len=256, pitch_shift_range=None, time_stretch_factors=None):
        self.data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                    self.data.append(json_data)
        self.seq_len = seq_len
        self.pitch_shift_range = pitch_shift_range or (-2, 3)
        self.time_stretch_factors = time_stretch_factors or [1, 1.05, 1.1]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        onset_events = self.data[idx]['onset_events']
        onset_events = self.pitch_shift(onset_events)
        onset_events = self.time_stretch(onset_events)
    
    # Add this block of code to check and adjust the data if necessary
        for event in onset_events:
            if event[1] < 21:
                event[1] = 21
            elif event[1] >= 108:  # 修改这里，使用 >= 而不是 >
                event[1] = 107  # 修改这里，将最大值设为 107

        words = onseteventstoword({'onset_events': onset_events})
        word_ints = wordtoint(words)
        input_word_ints = [pad_token] + word_ints[:-1]
        while len(input_word_ints) < self.seq_len:
            input_word_ints.append(pad_token)
        return input_word_ints[:self.seq_len], word_ints[:self.seq_len]



    def pitch_shift(self, onset_events):
        shift = random.randint(self.pitch_shift_range[0], self.pitch_shift_range[1])
        shifted_onset_events = [[event[0], event[1] + shift, event[2], event[3]] for event in onset_events]
        return shifted_onset_events

    def time_stretch(self, onset_events):
        factor = random.choice(self.time_stretch_factors)
        stretched_onset_events = [[round(event[0] * factor), event[1], event[2], round(event[3] * factor)] for event in onset_events]
        return stretched_onset_events


    def to_json(self, word_ints, output_file):
        words = inttoword(word_ints)
        pianoroll = wordtoonsetevents(words)
        with open(output_file, 'w') as f:
            json.dump(pianoroll, f)

    def save_preprocessed_data(self, output_file):
        input_data, target_data = [], []
        for i in range(len(self.data)):
            input_word_ints, target_word_ints = self.__getitem__(i)
            input_data.append(input_word_ints)
            target_data.append(target_word_ints)
        input_tensor = torch.tensor(input_data, dtype=torch.long)
        target_tensor = torch.tensor(target_data, dtype=torch.long)
        torch.save((input_tensor, target_tensor), output_file)

folder_path = "C:/Users/z7913/Documents/project_real/pianoroll"  # 请替换为你的文件夹路径
output_file = "preprocessed_data.pt"  # 保存预处理数据的文件名
dataset = PianoRollDataset(folder_path)
dataset.save_preprocessed_data(output_file)