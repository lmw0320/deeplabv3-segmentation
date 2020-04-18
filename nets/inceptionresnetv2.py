import os, sys
sys.path.append(os.path.dirname(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

from keras.layers import Conv2D, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Concatenate, Input
from keras.models import Model
import keras.backend as K

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
                kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                name=name)(x)
    if not use_bias:
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(scale=False,
                                name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x

def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    mixed = Concatenate(name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[-1],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=K.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x

def InceptionResNetV2(img_input):
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='same')
    x0  = x

    x = conv2d_bn(x, 32, 3, padding='same')
    x = conv2d_bn(x, 64, 3, padding='same')
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    x1  = x

    x = conv2d_bn(x, 80, 1, padding='same')
    x = conv2d_bn(x, 192, 3, padding='same')
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    x2  = x

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1, padding='same')
    branch_1 = conv2d_bn(x, 48, 1, padding='same')
    branch_1 = conv2d_bn(branch_1, 64, 5, padding='same')
    branch_2 = conv2d_bn(x, 64, 1, padding='same')
    branch_2 = conv2d_bn(branch_2, 96, 3, padding='same')
    branch_2 = conv2d_bn(branch_2, 96, 3, padding='same')
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, padding='same')
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(name='mixed_5b')(branches)
    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 5):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='same')
    branch_1 = conv2d_bn(x, 256, 1, padding='same')
    branch_1 = conv2d_bn(branch_1, 256, 3, padding='same')
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='same')
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(name='mixed_6a')(branches)
    x3  = x

    for block_idx in range(1, 5):
        x = inception_resnet_block(x, scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)
    
    branch_0 = conv2d_bn(x, 256, 1, padding='same')
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='same')
    branch_1 = conv2d_bn(x, 256, 1, padding='same')
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='same')
    branch_2 = conv2d_bn(x, 256, 1, padding='same')
    branch_2 = conv2d_bn(branch_2, 288, 3, padding='same')
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='same')
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(name='mixed_7a')(branches)
    for block_idx in range(1, 5):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx= block_idx) 
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)
    x4  = x

    return [x0, x1, x2, x3, x4]
if __name__ == "__main__":
    img_input = Input(shape=(256,256,3))
    levels = InceptionResNetV2(img_input)
    for i in range(len(levels)):
        print('%s.shape'%levels[i].name, levels[i].shape)