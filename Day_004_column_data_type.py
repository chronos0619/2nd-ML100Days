import os
import numpy as np
import pandas as pd

#設定datapath
dir_data = './data/'
f_app_train = os.path.join(dir_data, 'application_train.csv')
f_app_test = os.path.join(dir_data, 'application_test.csv')

app_train = pd.read_csv(f_app_train)
app_test = pd.read_csv(f_app_test)

#檢視資料中各個欄位類型的數量
print(app_train.dtypes.value_counts(), '\n')
#檢視資料中類別型欄位各自類別的數量
print(app_train.select_dtypes(include = ["object"]).apply(pd.Series.nunique, axis = 0), '\n')

#LabelEncoding
#import sklearn 
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le_count = 0

#Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        #If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            #Train on the training data
            le.fit(app_train[col])
            #Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col]  = le.transform(app_test[col])

            #Keep track of how many columns were label encoded
            le_count += 1

print('%d columns were label encoded.' % le_count, '\n')

#One Hot Encoding 

app_train = pd.get_dummies(app_train)
app_test  = pd.get_dummies(app_test)

print(app_train['CODE_GENDER_F'].head())
print(app_train['CODE_GENDER_M'].head())
print(app_train['NAME_EDUCATION_TYPE_Academic degree'].head())

app_train = pd.read_csv(f_app_train)
sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
print(sub_train.shape, '\n')
sub_train.head()