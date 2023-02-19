import os
import json

# 取得當前檔案的路徑
current_path = os.path.abspath(os.path.dirname(__file__))

# 讀取JSON檔案
with open('C://Users//謝向嶸//Desktop//專題用//PIANO//pianoroll.json', 'r') as f:
    data = json.load(f)
# 處理音符數據
notes = []
for note in data['notes']:
    # 計算音符持續時間
    duration = note['duration'] * 0.01
    # 縮小音符長度的範圍
    if duration > 4:
        duration = 4
    elif duration < 0.25:
        duration = 0.25
    # 將音符編碼為數字
    pitch = note['pitch']
    encoded_note = (pitch, duration)
    # 去除重複的音符
    if encoded_note not in notes:
        notes.append(encoded_note)

# 將音符保存到文件中
with open('C://Users//謝向嶸//Desktop//專題用//PIANO//processed_notes.txt', 'w') as f:
    for note in notes:
        f.write(f'{note[0]} {note[1]}\n')
