from torch.utils.data import Dataset
import os
import json
import time

<<<<<<< HEAD
=======

>>>>>>> origin/master
class json_list(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
<<<<<<< HEAD
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
=======
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
>>>>>>> origin/master


root_dir = "C:\\Users\\謝向嶸\\Desktop\\專題用\\PIANO"
label_dir = "pianoroll"
pr_dataset = json_list(root_dir, label_dir)

for i, (data, label) in enumerate(pr_dataset):
    print(f"Item {i}:")
    print(f"Data: {data}")
    print(f"Label: {label}")
<<<<<<< HEAD
    # time.sleep(0.1)
=======
    time.sleep(0.1)
>>>>>>> origin/master
