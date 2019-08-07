import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import randn

dir_data = './data/'
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)
s = pd.Series([1, 2, 3])
s.describe()
plt.hist(randn(100), bins=20, color='k', alpha=0.3)
plt.show()