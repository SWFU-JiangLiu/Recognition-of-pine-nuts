import os
import shutil
import pandas as pd
import numpy as np

def getFileList(path):
    filenames_set =set()
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            print(file)
            filenames_set.add(dirpath + u'\\' + file)
    return filenames_set

if __name__ == '__main__':
    addressPDF = u'./../data/NIR_data/pinus_raw_data'
    path=getFileList(addressPDF)
    # target=['K326','NC297','NC102']
    lable=0
    new=np.ones(2336)
    print(path)
    for i in path:
        df=pd.read_csv(i,header=None)
        # df=pd.read_excel(i,header=None)
        # df=df.dropna(axis=0)
        # print(df.info())
        df = pd.DataFrame(df)
        DF_Y=df[1]
        DF_T=DF_Y.T   #转置
        new_DF_T=DF_T.tolist()
        # print(type(new_DF_T))
        if i.find('云南松')>=0:
            lable=1
        if i.find('华山松')>=0:
            lable=2
        if i.find('油松')>=0:
            lable=3
        if i.find('湿地')>=0:
            lable=4
        if i.find('白皮松')>=0:
            lable=5
        if i.find('马尾')>=0:
            lable=6
        if i.find('黑松')>=0:
            lable=7
        print(i)
        new_DF_T.append(lable)
        df_lable=np.array(new_DF_T)
        new = np.vstack((new, df_lable))
    np.savetxt('./../data/NIR_data/1_Pinus_NIR_data_all_with_lable.csv', new, delimiter=',')
