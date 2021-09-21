import os
import cv2
import numpy as np
if __name__ == '__main__':
    path = 'E:\jiang\data\pinus/all/'  # 源文件夹
    savepath='E:\jiang\data\pinus\improve/'
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
            # h_flip,v_flip,hv_flip,resize_60=imagedeal(filepath, save_dir_path)
            img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
            # 水平镜像
            h_flip = cv2.flip(img, 1)
            # 垂直镜像
            v_flip = cv2.flip(img, 0)
            # 水平垂直镜像
            hv_flip = cv2.flip(img, -1)
            # 缩放60%
            resize_60 = cv2.resize(img, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_NEAREST)
            filesavepath = save_dir_path + file_obj
            cv2.imencode('.jpg', img)[1].tofile(filesavepath)

            filesavepath = save_dir_path + 'h_flip_'+file_obj
            cv2.imencode('.jpg', h_flip)[1].tofile(filesavepath)

            filesavepath = save_dir_path + 'v_flip_'+file_obj
            cv2.imencode('.jpg', v_flip)[1].tofile(filesavepath)

            filesavepath = save_dir_path + 'hv_flip_'+file_obj
            cv2.imencode('.jpg', hv_flip)[1].tofile(filesavepath)

            filesavepath = save_dir_path + 'resize_60_'+file_obj
            cv2.imencode('.jpg', resize_60)[1].tofile(filesavepath)



