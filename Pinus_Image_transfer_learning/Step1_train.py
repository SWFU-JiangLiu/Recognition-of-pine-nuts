from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, data, color, transform
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import imagenet_utils, Xception
from tensorflow.keras.optimizers import Adam
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow. keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
# 绘loss
def loss_plot(file,hist, title):
    plt.figure(dpi=600)
    plt.plot(np.arange(len(hist.history['loss'])), hist.history['loss'], label='training')
    plt.plot(np.arange(len(hist.history['val_loss'])), hist.history['val_loss'], label='validation')
    plt.title(file+'_'+title + ' Train and Validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc=0)
    img_title = file+'_loss' + title + '.png'
    plt.savefig(img_title)
    plt.show()
    return 0
# 绘制acc
def acc_plot(file,hist, title,flag):
    plt.figure(dpi=600)
    val_flag='val_'+flag
    plt.plot(np.arange(len(hist.history[flag])), hist.history[flag], label='training')
    plt.plot(np.arange(len(hist.history[val_flag])), hist.history[val_flag], label='validation')
    plt.title(file+' '+title + ' Train and Validation')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc=0)
    img_title = file+'_acc' + title + '.png'
    plt.savefig(img_title)
    plt.show()
    return 0
# InceptionV3模型
def InceptionV3_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=197, img_cols=197, RGB=True,
                      is_plot_model=False):
    color = 3 if RGB else 1
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling=None,
                             input_shape=(img_rows, img_cols, color),
                             classes=nb_classes)

    # 冻结base_model所有层，这样就可以正确获得bottleneck特征
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    # 添加自己的全链接分类层
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    # 训练模型
    model = Model(inputs=base_model.input, outputs=predictions)
    sgd = tf.keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    # # 绘图
    # if is_plot_model:
    #     plot_model(model, to_file='inception_v3_model.png', show_shapes=True)

    return model
# Vgg16模型
def Vgg16_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=197, img_cols=197, RGB=True,
                      is_plot_model=False):
    color = 3 if RGB else 1
    base_model = VGG16(weights='imagenet', include_top=False, pooling=None,
                             input_shape=(img_rows, img_cols, color),
                             classes=nb_classes)

    # 冻结base_model所有层，这样就可以正确获得bottleneck特征
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    # 添加自己的全链接分类层
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    # 训练模型
    model = Model(inputs=base_model.input, outputs=predictions)
    sgd = tf.keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # # 绘图
    # if is_plot_model:
    #     plot_model(model, to_file='inception_v3_model.png', show_shapes=True)

    return model
# Vgg19模型
def Vgg19_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=197, img_cols=197, RGB=True,
                      is_plot_model=False):
    color = 3 if RGB else 1
    base_model = VGG19(weights='imagenet', include_top=False, pooling=None,
                             input_shape=(img_rows, img_cols, color),
                             classes=nb_classes)

    # 冻结base_model所有层，这样就可以正确获得bottleneck特征
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    # 添加自己的全链接分类层
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    # 训练模型
    model = Model(inputs=base_model.input, outputs=predictions)
    sgd = tf.keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model
# Resnet50模型
def Resnet50_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=197, img_cols=197, RGB=True,
                      is_plot_model=False):
    color = 3 if RGB else 1
    base_model = ResNet50(weights='imagenet', include_top=False, pooling=None,
                             input_shape=(img_rows, img_cols, color),
                             classes=nb_classes)

    # 冻结base_model所有层，这样就可以正确获得bottleneck特征
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    # 添加自己的全链接分类层
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    # 训练模型
    model = Model(inputs=base_model.input, outputs=predictions)
    sgd = tf.keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model
# Xception模型
def Xception_model(self, lr=0.005, decay=1e-6, momentum=0.9,nb_classes=2, img_rows=197, img_cols=197, RGB=True,
                      is_plot_model=False):
    color = 3 if RGB else 1
    base_model = Xception(weights='imagenet', include_top=False, pooling=None,
                             input_shape=(img_rows, img_cols, color),
                             classes=nb_classes)

    # 冻结base_model所有层，这样就可以正确获得bottleneck特征
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    # 添加自己的全链接分类层
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    # 训练模型
    model = Model(inputs=base_model.input, outputs=predictions)
    sgd = tf.keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model
# 数据准备
def DataGen(dir_path, img_row, img_col, batch_size, is_train):
    if is_train:
        datagen = ImageDataGenerator(rescale=1. / 255,
                                     zoom_range=0.25, rotation_range=15.,
                                     channel_shift_range=25., width_shift_range=0.02, height_shift_range=0.02,
                                     horizontal_flip=True, fill_mode='constant')
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        dir_path, target_size=(img_row, img_col),
        batch_size=batch_size,
        # class_mode='binary',
        shuffle=is_train)

    return generator


if __name__ == '__main__':
    n_classes=7
    image_size = 224
    batch_size = 64
    epochs=100
    data_path=['E:/jiang/data/pinus/pinus_split_data_folser']
    # data_path=['E:/jiang/data/pinus/improve_pinus_split_data_folser']
    # data_path=['E:/jiang/data/pinus/improve_gray_pinus_split_data_folser']
    for path in data_path:
        files=path.split('/')
        # file=files[3]
        file='pinus'
        print(file)
        print(path)
        train_dir =path+'/train'
        val_dir = path+'/val'
        train_generator = DataGen(train_dir, image_size, image_size, batch_size, True)
        validation_generator = DataGen(val_dir, image_size, image_size, batch_size, False)
        # 调用模型
        vgg16_model = Vgg16_model(validation_generator,nb_classes=n_classes, img_rows=image_size, img_cols=image_size)
        vgg19_model = Vgg19_model(validation_generator,nb_classes=n_classes, img_rows=image_size, img_cols=image_size)
        xception_model = Xception_model(validation_generator,nb_classes=n_classes, img_rows=image_size, img_cols=image_size)
        resnet50_model = Resnet50_model(validation_generator,nb_classes=n_classes, img_rows=image_size, img_cols=image_size)
        inceptionv3_model = InceptionV3_model(validation_generator,nb_classes=n_classes, img_rows=image_size, img_cols=image_size)
        net = [vgg16_model, vgg19_model, xception_model, resnet50_model, inceptionv3_model]
        title_list = ['vgg16_model', 'vgg19_model', 'xception_model', 'resnet50_model', 'inceptionv3_model']
        # net = [vgg16_model]
        # title_list = ['vgg16_model']
        result=np.zeros(2)
        for i in range(len(net)):
            hist = net[i].fit_generator(
                train_generator,
                # steps_per_epoch=100,
                epochs=epochs,
                validation_data=validation_generator,
                # validation_steps=2
            )
            # print(hist.history['acc'],hist.history['val_acc'])
            print(net[i].evaluate_generator(validation_generator))
            result=np.vstack((result,net[i].evaluate_generator(validation_generator)))
            title = title_list[i]
            print(title)
            print(result)
            flag='accuracy'
            if title=='inceptionv3_model':
                flag='acc'
            acc_plot(file,hist,title,flag)
            loss_plot(file,hist,title)
        save_name=file+' result.txt'
        print(save_name)
        np.savetxt(save_name, result, delimiter='.')