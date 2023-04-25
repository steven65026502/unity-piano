import os
import json
from dictionary_roll import onseteventstoword, wordtoonsetevents, wordtoint, inttoword


def load_data(data_path):
    data_files = os.listdir(data_path)
    data = []
    max_len = 2048  # Maximum sequence length

    for file in data_files:
        file_path = os.path.join(data_path, file)

        with open(file_path, "r") as f:
            pianoroll = json.load(f)
            words = onseteventstoword(pianoroll)
            word_ints = wordtoint(words)
            if len(word_ints) <= max_len:
                # Pad the sequence with zeros to ensure a fixed length
                word_ints += [0] * (max_len - len(word_ints))
                data.append(word_ints)

    return data

def load_test_data(test_data_path):
    test_data_files = os.listdir(test_data_path)
    test_data = []
    test_labels = []  # 这个列表将存储测试数据的真实标签
    max_len = 2048  # Maximum sequence length

    for file in test_data_files:
        file_path = os.path.join(test_data_path, file)

        with open(file_path, "r") as f:
            pianoroll = json.load(f)
            words = onseteventstoword(pianoroll)
            word_ints = wordtoint(words)
            if len(word_ints) <= max_len:
                # Pad the sequence with zeros to ensure a fixed length
                word_ints += [0] * (max_len - len(word_ints))
                test_data.append(word_ints)

    return test_data, test_labels