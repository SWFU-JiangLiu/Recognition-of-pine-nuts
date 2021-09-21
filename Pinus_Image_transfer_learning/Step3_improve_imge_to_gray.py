import os
import cv2
import numpy as np
if __name__ == '__main__':
    path = 'E:\jiang\data\pinus\improve/'  # 源文件夹
    savepath='E:\jiang\data\pinus\improve_gray/'
    keys = ['1_白皮松_baipisong','2_云南松_yunnansong','3_黑松_heisong','4_华山松_huashansong','5_马尾松_maweisong','6_湿地松_shidisong','7_油松_yousong']
    for i in keys:
        key = i
        src_dir_path = path + key + '/'
        save_dir_path=savepath+ key + '/'
        if not os.path.exists(save_dir_path):
            print("save_dir_path not exist,so create the dir")
            os.mkdir(save_dir_path, 1)
        if os.path.exists(src_dir_path):
            print("src_dir_path exitst")
        # 返回指定的文件夹包含的文件或文件夹的名字的列表
        file_list = os.listdir(src_dir_path)
        for file_obj in file_list:
            filepath=src_dir_path+file_obj
            img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            filesavepath = save_dir_path + file_obj
            cv2.imencode('.jpg', img_gray)[1].tofile(filesavepath)


