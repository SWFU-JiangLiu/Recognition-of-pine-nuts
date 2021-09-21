import pandas as pd
import numpy as np
df = pd.read_csv('./../data/NIR_data/2_Pinus_NIR_data_all_with_lable_add_index.csv')
x= df.drop('class', axis=1)
y = df['class']
x_row = len(x)
x_norm3 = np.empty((0, x_row), int)
for j in range(len(x.T)):
    x_li = x.iloc[:, j]
    x_No = (x_li - x_li.min()) / (x_li.max() - x_li.min())
    x_No = np.array(x_No)
    x_norm3 = np.vstack((x_norm3, x_No.T))
x_norm3 = x_norm3.T
x_norm3 = np.array(x_norm3)
print(x_norm3.shape)
y= np.array(y.T)
y=y.reshape(209,1)
print(y.shape)
all=np.hstack((x_norm3, y))
print(all.shape)
np.savetxt('./../data/NIR_data/3_Pinus_nominal_NIR_data_all.csv', all, delimiter=',')