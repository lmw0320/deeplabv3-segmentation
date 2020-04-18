import sys, os
sys.path.append(os.path.dirname(__file__)) #临时加入环境变量，以便下面代码导入模块
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

from math import ceil
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.layers import Softmax,Reshape, Concatenate, Add
from keras.models import Model

from nets.efficientnet import EfficientNetB0, EfficientNetB2
from nets.inceptionresnetv2 import InceptionResNetV2
from nets.mobilenetv2 import mobilenetV2
from nets.mobilenetv3 import mobilenetV3
from nets.new_modelv1 import New_ModelV1


def interp_block(prev_layer, level, feature_map_size):
    pool_size = strides = (int(feature_map_size[0] /  level), int(feature_map_size[1] /  level))                     
    prev_layer = AveragePooling2D(pool_size, strides=strides, padding = 'same')(prev_layer)
    prev_layer = Conv2D(512, 1, strides=1, use_bias=False)(prev_layer)
    prev_layer = BatchNormalization()(prev_layer)
    prev_layer = Activation('relu')(prev_layer)  #进行均值池化后，得到1*1， 2*2， 3*3，6*6大小的特征图，然后卷积成512通道的
    #keras.layers中没有的方法，必须使用lambda，确保各层的连续性
    #tf.image.resize_images中，method =0 代表双线性插值，1代表最近邻，2代表双三次插值，3代表面积插值法
    prev_layer = Lambda(lambda xx: tf.image.resize(xx, feature_map_size[:2], method = 0, align_corners= True))(prev_layer)
    return prev_layer

def PSPnet(input_shape, classes, bone_name):
    img_input = Input(shape =input_shape)
    #利用主干网络实现特征提取
    if bone_name == "efficientnet_b0": 
        layers = EfficientNetB0(img_input)

    if bone_name == "efficientnet_b2":
        layers = EfficientNetB2(img_input)

    if bone_name == "inceptionresnetv2":
        layers = InceptionResNetV2(img_input)

    if bone_name == "mobilenetv2":
        layers = mobilenetV2(img_input)

    if bone_name == "mobilenetv3":
        layers = mobilenetV3(img_input)

    if bone_name == "new_modelv1":
        layers = New_ModelV1(img_input)
  
    # ---PSPNet concat layers with Interpolation
    levels = [1, 2, 3, 6]
    layers = list(layers)
    layers.reverse() #layers为浅层到深层的特征图，大小为逐渐变小，因此考虑反向处理，逐步放大，有些类似Unet结构
    for x in layers:
        # print('ori_x.shape', x.shape)
        feature_map_size = K.int_shape(x)[1:3]
        #interp_block会统一放大成原始输入大小，通道数为512的特征图。
        interp_block1 = interp_block(x, levels[0], feature_map_size)
        interp_block2 = interp_block(x, levels[1], feature_map_size)
        interp_block3 = interp_block(x, levels[2], feature_map_size)
        interp_block6 = interp_block(x, levels[3], feature_map_size)
        x = Concatenate()([x,
                        interp_block6,
                        interp_block3,
                        interp_block2,
                        interp_block1])
        # print('x.shape after concate', x.shape)
    x = Conv2D(512, (3, 3), strides=1, padding="same", use_bias=False)(x)
    x = BatchNormalization(momentum=0.95, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(classes, 1, padding = 'same')(x)        
    x = Lambda(lambda xx: tf.image.resize(xx,input_shape[:2]))(x)

    x = Reshape((-1, classes))(x)
    x = Softmax()(x)
    model = Model(img_input, x, name='pspnet')

    return model

if __name__ == "__main__":
    # bone_name: ["new_modelv1","efficientnet_b0", "efficientnet_b2", "mobilenetv3", "mobilenetv2", "inceptionresnetv2"]
    bone_name = 'efficientnet_b0'
    model = PSPnet(input_shape=(256, 256, 3), classes=5, bone_name = bone_name)
    model.summary()
    print(bone_name, "--", model.name)