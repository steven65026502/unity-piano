import torch
import pickle


class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, inputs_file, targets_file):
        # Load inputs and targets from disk
        with open(inputs_file, 'rb') as f:
            self.inputs = pickle.load(f)
        with open(targets_file, 'rb') as f:
            self.targets = pickle.load(f)

        # Get the number of unique tokens in the dataset
        self.num_tokens = len(set(self.inputs.flatten().tolist() + self.targets.tolist()))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return torch.LongTensor(self.inputs[index]), torch.LongTensor(self.targets[index])
