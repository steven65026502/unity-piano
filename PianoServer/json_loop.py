from torch.utils.data import Dataset
import os
import json
import time


class json_list(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.json_path = os.listdir(self.path)

    def __getitem__(self, idx):
        json_name = self.json_path[idx]
        json_item_path = os.path.join(self.root_dir, self.label_dir, json_name)
        with open(json_item_path) as f:
            data = json.load(f)
        label = label_dir
        return data, label

    def __len__(self):
        return len(self.json_path)


root_dir = "C:\\Users\\謝向嶸\\Desktop\\專題用\\PIANO"
label_dir = "pianoroll"
pr_dataset = json_list(root_dir, label_dir)

for i, (data, label) in enumerate(pr_dataset):
    print(f"Item {i}:")
    print(f"Data: {data}")
    print(f"Label: {label}")
    time.sleep(0.1)
