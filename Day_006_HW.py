import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir_data = './data/'

f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()
print(app_train.head(), '\n')

#(app_train['DAYS_BIRTH'] / (-365)).describe()
# DAYS_BIRTH:申請貸款時的年齡
print((app_train['DAYS_BIRTH'] / (-365)).describe(), '\n')

#DAYS_EMPLOYED: 申請貸款前,申請人已在現職工作的時間
(app_train['DAYS_EMPLOYED'] / 365).describe()
plt.hist(app_train['DAYS_EMPLOYED'])
plt.show()
print(app_train['DAYS_EMPLOYED'].value_counts(), '\n')

#排除異常值
anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom), '\n')

print(sum(app_train['DAYS_EMPLOYED'] == 365243) / len(app_train), '\n')

app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
print(app_train['DAYS_EMPLOYED_ANOM'].value_counts(), '\n')

#用 nan 取代異常值
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')
plt.show()

#OWN_CAR_AGE: 貸款人的車齡
plt.hist(app_train[~app_train.OWN_CAR_AGE.isnull()]['OWN_CAR_AGE'])
plt.show()
print(app_train['OWN_CAR_AGE'].value_counts(), '\n')

#統計並印出車齡大於五十的資料
print(app_train[app_train['OWN_CAR_AGE'] > 50]['OWN_CAR_AGE'].value_counts(), '\n')

print("Target of OWN_CAR_AGE >= 50: %.2f%%" % (app_train[app_train['OWN_CAR_AGE'] >= 50]['TARGET'].mean() * 100))
print("Target of OWN_CAR_AGE < 50: %.2f%%" % (app_train[app_train['OWN_CAR_AGE'] < 50]['TARGET'].mean() * 100))

app_train['OWN_CAR_AGE_ANOM'] = app_train['OWN_CAR_AGE'] >= 50