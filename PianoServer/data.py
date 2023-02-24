import json
import os
from PianoServer import dictionary_roll

data = []
path = "C:\\Users\\謝向嶸\\Desktop\\專題用\\PIANO\\pianoroll"
dir = os.listdir(path)

for filename in dir:
    if filename.endswith('.json'):
        with open(os.path.join(path, filename), 'r') as f:
            json_data = json.load(f)
        data.append(json_data)

word_data = []
for item in data:
    word_item = dictionary_roll.onseteventstoword(item)
    word_data.append(word_item)

wordint_data = []
for item in word_data:
    wordint_item = dictionary_roll.wordtoint(item)
    wordint_data.append(wordint_item)

# 將 wordint_data 寫入到 JSON 檔案中
with open('wordint_data.json', 'w') as f:
    json.dump(wordint_data, f)

# 從 JSON 檔案中載入 wordint_data
with open('wordint_data.json', 'r') as f:
    wordint_data_loaded = json.load(f)

# 確認載入的 wordint_data 是否跟原來的一樣
print(wordint_data == wordint_data_loaded)
