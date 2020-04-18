import sys, os
path = os.path.dirname(__file__)
sys.path.append(path) #临时加入环境变量，以便下面代码导入模块
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
from keras import layers
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Activation, BatchNormalization, Input, DepthwiseConv2D

def maxpool_bn(inputs, filters):
    x1_1 = MaxPooling2D(pool_size = 2, strides = 1, padding = 'same')(inputs)
    x1_2 = MaxPooling2D(pool_size = 3, strides = 1, padding = 'same')(inputs)
    x1_3 = MaxPooling2D(pool_size = 4, strides = 1, padding = 'same')(inputs)
    x1_4 = MaxPooling2D(pool_size = 5, strides = 1, padding = 'same')(inputs)
    x1_5 = MaxPooling2D(pool_size = 6, strides = 1, padding = 'same')(inputs)
    x1_6 = MaxPooling2D(pool_size = 7, strides = 1, padding = 'same')(inputs)

    x = Concatenate()([x1_1, x1_2, x1_3, x1_4, x1_5, x1_6]) #将不同大小的目标物体的边缘信息，加入到特征中
    # x = Concatenate()([x1_2, x1_4, x1_6])
    #训练时，个人以为最好使用2*2卷积，步长为2，既可以压缩大小，又能尽量吸收信息。
    x = Conv2D(filters, kernel_size = 1, strides = 2, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x
#-----------------------------------------------------------------------------------------------------------------------#
#主网络部分

def  New_ModelV1(inputs):
    basic_filter = 128
    temp = []
    x = inputs
    for i in range(5): #设置5个阶段的网络加深处理。个人觉得通道数不应该减少，否则容易丢失信息，虽然这样会增加一定的参数量。
        x = maxpool_bn(x, basic_filter * (i+1))
        temp.append(x)
    return temp

if __name__ == "__main__":
    img_input = Input(shape = (256, 256 ,3))
    levels = New_ModelV1(img_input)
    for i in range(len(levels)):
        print('%s.shape'%levels[i].name, levels[i].shape)