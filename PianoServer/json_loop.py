from torch.utils.data import Dataset
import os
import json
import time

class json_list(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.num_files = len(os.listdir(self.path))

    def __getitem__(self, idx):
        if idx >= self.num_files:
            raise IndexError
        json_name = f'{idx}.json'
        json_item_path = os.path.join(self.root_dir, self.label_dir, json_name)
        with open(json_item_path) as f:
            data = json.load(f)
        label = self.label_dir
        return data, label

    def __len__(self):
        return self.num_files


root_dir = "C:\\Users\\謝向嶸\\Desktop\\專題用\\PIANO"
label_dir = "pianoroll"
pr_dataset = json_list(root_dir, label_dir)

for i, (data, label) in enumerate(pr_dataset):
    print(f"Item {i}:")
    print(f"Data: {data}")
    print(f"Label: {label}")
    # time.sleep(0.1)
