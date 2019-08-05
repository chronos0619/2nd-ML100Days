import os
import pandas as pd
import numpy as np
import json
import pickle

#一般讀取並輸出
with open("data/example.txt", 'r') as f:
    data = f.readlines()
print(data) 
print('\n')

#將 txt 轉成 pandas dataframe
data = []

with open("data/example.txt", 'r') as f:
    for line in f:
        line = line.replace('\n', '').split(',')
        # 將每句最後的 /n 取代成空值後，再以逗號斷句
        data.append(line)

print(data)
print('\n')

#依照行的形式輸出資料
df = pd.DataFrame(data[1:])
df.columns = data[0]
print(df)
print('\n')

#將資料轉成 json 檔後輸出
df.to_json('data/example01.json')

with open('data/example01.json', 'r') as f:
    j1 = json.load(f)

print(j1)
print('\n')

# Set the index to become the 'id' column:
df.set_index('id', inplace=True)
print(df)
print('\n')

df.to_json('data/example02.json', orient = 'index')

with open('data/example02.json', 'r') as f:
    j2 = json.load(f)

print(j2)
print('\n')

#將檔案存為 npy 檔
array = np.array(data[1:])
print(array)
print("\n")

np.save(arr = array, file='data/example.npy')

array_back = np.load('data/example.npy')
print(array_back)
print("\n")

#Pickle
with open('data/example.pkl', 'wb') as f:
    pickle.dump(file = f, obj = data)

with open('data/example.pkl', 'rb') as f:
    pkl_data = pickle.load(f)
print(pkl_data)
print('\n')





