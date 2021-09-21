import os

def split_foler_f(fd1):
    p1, name = os.path.split(fd1)
    # import split_folders
    import splitfolders
    splitfolders.ratio(fd1, output=os.path.join(p1, "WT_split_data_folser"), seed=1337,
                        ratio=(.8, .2))  # default values
    print('Split Done！')

if __name__ == '__main__':
    foler_path_list=['E:\jiang\data\Folk_music\WT']
    for folder_path in foler_path_list:
        split_foler_f(folder_path)