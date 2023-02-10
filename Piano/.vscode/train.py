import json
for i in range(10):
    path = f'C:\\Users\\謝向嶸\\20220928\\PIANO\\pianoroll\\{i}.json'
    with open(path) as f:
        data = json.load(f)
        print(data)
