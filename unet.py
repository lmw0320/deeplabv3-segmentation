import sys, os
sys.path.append(os.path.dirname(__file__)) #临时加入环境变量，以便下面代码导入模块
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import numpy as np
from keras.models import Model
from keras.layers import Conv2DTranspose, UpSampling2D, Conv2D, BatchNormalization, Activation, Concatenate, Input, Reshape

from nets.efficientnet import EfficientNetB0, EfficientNetB2
from nets.inceptionresnetv2 import InceptionResNetV2
from nets.mobilenetv2 import mobilenetV2
from nets.mobilenetv3 import mobilenetV3
from nets.new_modelv1 import New_ModelV1

def handle_block_names(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu', name=relu_name)(x)
        return x
    return layer

def Upsample2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                     use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)
        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)
        if skip is not None:
            x = Concatenate()([x, skip])
        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)
        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        return x
    return layer

def Transpose2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),
                      transpose_kernel_size=(4,4), use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                            padding='same', name=up_name, use_bias=not(use_batchnorm))(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name+'1')(x)
        x = Activation('relu', name=relu_name+'1')(x)

        if skip is not None:
            x = Concatenate()([x, skip])
        x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)

        return x
    return layer

def Unet(input_shape =(256, 256, 3),
        classes = 1,
        bone_name = 'resnet50',
        freeze_encoder = True,
        block_type ='upsampling'):

    img_input = Input(shape = input_shape)

    if bone_name == "new_modelv1":
        levels = New_ModelV1(img_input)
        [f1 , f2 , f3 , f4 , f5] = levels
        skip = [f4, f3, f2, f1, None] 
        x = f5
 
    if bone_name == "efficientnet_b0":
        levels = EfficientNetB0(img_input)
        [f1 , f2 , f3 , f4 , f5] = levels
        skip = [f4, f3, f2, f1, None]
        x = f5

    if bone_name == "efficientnet_b2":
        levels = EfficientNetB2(img_input)
        [f1 , f2 , f3 , f4 , f5] = levels
        skip = [f4, f3, f2, f1, None]
        x = f5

    if bone_name == "inceptionresnetv2":
        levels = InceptionResNetV2(img_input)
        [f1 , f2 , f3 , f4 , f5] = levels
        skip = [f4, f3, f2, f1, None]
        x = f5              

    if bone_name == "mobilenetv2":
        levels = mobilenetV2(img_input)
        [f1 , f2 , f3] = levels
        skip = [f2, f1, None]
        x = f3

    if bone_name == "mobilenetv3":
        levels = mobilenetV3(img_input)
        [f1 , f2 , f3 , f4 , f5] = levels
        skip = [f4, f3, f2, f1, None]
        x = f5
    if classes > 1:
        activation = "softmax"
    else:
        activation = "sigmoid"

    if block_type == 'transpose': #选择转置卷积或是上采样，来放大feature map
        up_block = Transpose2D_block
    elif block_type == 'upsampling':
        up_block = Upsample2D_block

    n_upsample_blocks = len(levels) #上采样层数, 主要看返回值有几个，才能获取几层。
    origin_filters = 256 #原始的卷积核数量
    for i in range(n_upsample_blocks):
        x = up_block(int(origin_filters / 2**i) , i, upsample_rate= 2,
                        use_batchnorm=True, skip = skip[i])(x)

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
    # x = Lambda(lambda xx: tf.image.resize(xx,input_shape[:2]))(x) #为了确保输出为原图大小
    x = Reshape((-1,classes))(x)
    x = Activation(activation, name=activation)(x)

    model = Model(img_input, x, name = 'Unet++')
    return model

if __name__ == "__main__":
    # bone_name: ["new_modelv1", "efficientnet_b0", "efficientnet_b2", "mobilenetv3", "mobilenetv2", "inceptionresnetv2"]
    bone_name = 'inceptionresnetv2'
    model = Unet(input_shape=(256, 256, 3), classes=5, bone_name = bone_name)
    model.summary()
    print(bone_name, "--", model.name)