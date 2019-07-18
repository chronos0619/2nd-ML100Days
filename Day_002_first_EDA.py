import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
 
dir_data = './data/'

f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)

#row數量以及column數量
print(app_train.shape)
#列出所有欄位
print(app_train.head())
#擷取部分資料
print(app_train.iloc[1,1])
print(app_train.iloc[0:2,:])
print(app_train.iloc[2:0,:])