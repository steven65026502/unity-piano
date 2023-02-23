import json
import os
from PianoServer.server import Dictionary

data = []
data_dir = 'pianoroll'

for filename in os.listdir(data_dir):
    print(filename)
    if filename.endswith('.json'):
        with open(os.path.join(data_dir, filename), 'r') as f:
            json_data = json.load(f)
        data.append(json_data)

        processed_data = []
        for item in json_data:
            processed_item = Dictionary.onseteventstoword(item)
            processed_data.append(processed_item)

merged_data = [item for sublist in data for item in sublist]


print(processed_data)