import sys, os
path = os.path.dirname(__file__)
sys.path.append(path) #临时加入环境变量，以便下面代码导入模块
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import math
from typing import List

from keras import backend as K
from keras.layers import Input, Multiply, Activation, Conv2D, Lambda, BatchNormalization, DepthwiseConv2D, Add
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, GlobalMaxPooling2D
from keras.models import Model
from keras.utils import get_file, get_source_inputs

from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input as _preprocess

from nets.utils.config import BlockArgs, get_default_block_list
from nets.utils.custom_objects import EfficientNetConvInitializer, EfficientNetDenseInitializer, Swish, DropConnect
def preprocess_input(x, data_format=None):
    return _preprocess(x, data_format, mode='torch', backend=K)

def round_filters(filters, width_coefficient, depth_divisor, min_depth):
    """Round number of filters based on depth multiplier."""
    multiplier = float(width_coefficient)
    divisor = int(depth_divisor)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    """Round number of filters based on depth multiplier."""
    multiplier = depth_coefficient

    if not multiplier:
        return repeats

    return int(math.ceil(multiplier * repeats))

def SEBlock(input_filters, se_ratio, expand_ratio, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

    num_reduced_filters = max(
        1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = Swish()(x)
        # Excite
        x = Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = Activation('sigmoid')(x)
        out = Multiply()([x, inputs])
        return out

    return block

def MBConvBlock(input_filters, output_filters,
                kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                data_format=None):

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio

    def block(inputs):

        if expand_ratio != 1:
            x = Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=EfficientNetConvInitializer(),
                padding='same',
                use_bias=False)(inputs)
            x = BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon)(x)
            x = Swish()(x)
        else:
            x = inputs

        x = DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=strides,
            depthwise_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = Swish()(x)

        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio,
                        data_format)(x)

        # output phase

        x = Conv2D(
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):

                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = DropConnect(drop_connect_rate)(x)

                x = Add()([x, inputs])

        return x

    return block

def EfficientNet(input_tensor,
                 block_args_list: List[BlockArgs],
                 width_coefficient: float,
                 depth_coefficient: float,
                 include_top=True,
                 weights=None,
                 pooling=None,
                 classes=1000,
                 dropout_rate=0.,
                 drop_connect_rate=0.,
                 batch_norm_momentum=0.99,
                 batch_norm_epsilon=1e-3,
                 depth_divisor=8,
                 min_depth=None,
                 data_format=None,
                 default_size=None,
                 **kwargs):
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if default_size is None:
        default_size = 224

    if block_args_list is None:
        block_args_list = get_default_block_list()
        
    # count number of strides to compute min size
    stride_count = 1
    for block_args in block_args_list:
        if block_args.strides is not None and block_args.strides[0] > 1:
            stride_count += 1

    min_size = int(2 ** stride_count)

    # # Determine proper input shape and default size.
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=default_size,
    #                                   min_size=min_size,
    #                                   data_format=data_format,
    #                                   require_flatten=include_top,
    #                                   weights=weights)

    # # Stem part
    # if input_tensor is None:
    #     inputs = layers.Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         inputs = layers.Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         inputs = input_tensor

    # x = inputs
    x = Conv2D(
        filters=round_filters(32, width_coefficient,
                              depth_divisor, min_depth),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=EfficientNetConvInitializer(),
        padding='same',
        use_bias=False)(input_tensor)
    x = BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = Swish()(x)
    x0 = x
    num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

    # Blocks part
    temp = []
    for block_idx, block_args in enumerate(block_args_list):
        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args.input_filters = round_filters(block_args.input_filters, width_coefficient, depth_divisor, min_depth)
        block_args.output_filters = round_filters(block_args.output_filters, width_coefficient, depth_divisor, min_depth)
        block_args.num_repeat = round_repeats(block_args.num_repeat, depth_coefficient)

        # The first block needs to take care of stride and filter size increase.
        x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                        block_args.kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                        batch_norm_momentum, batch_norm_epsilon, data_format)(x)
        temp.append(x)
        if block_args.num_repeat > 1:
            block_args.input_filters = block_args.output_filters
            block_args.strides = [1, 1]

        for _ in range(block_args.num_repeat - 1):
            x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                            block_args.kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                            batch_norm_momentum, batch_norm_epsilon, data_format)(x)
  
    x1 = temp[1]
    x2 = temp[2]
    x3 = temp[4]
    # Head part
    #输出是输入是1/32大小
    x = Conv2D(
        filters=round_filters(1280, width_coefficient, depth_coefficient, min_depth),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=EfficientNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    #增加通道数到1280，大小不变
    x = BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = Swish()(x) #输出是输入的1/32大小
    
    if include_top:
        x = GlobalAveragePooling2D(data_format=data_format)(x)

        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

        x = Dense(classes, kernel_initializer=EfficientNetDenseInitializer())(x)
        x = Activation('softmax')(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    # 输出为输入图片大小的1/32
    x4 = x
    

    return x0, x1, x2, x3, x4

def EfficientNetB0(input_tensor,
                   include_top=False,
                   weights='imagenet',
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.2,
                   drop_connect_rate=0.,
                   data_format='channel_last'):
 
    return EfficientNet(input_tensor,
                        get_default_block_list(),
                        width_coefficient=1.0,
                        depth_coefficient=1.0,
                        include_top=include_top,
                        weights=weights,
                        pooling=pooling,
                        classes=classes,
                        dropout_rate=dropout_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=256)

def EfficientNetB2(input_tensor,
                   include_top=False,
                   weights='imagenet',
                   pooling=None,
                   classes=1000,
                   dropout_rate=0.3,
                   drop_connect_rate=0.,
                   data_format='channel_last'):

    return EfficientNet(input_tensor,
                    get_default_block_list(),
                    width_coefficient=1.1,
                    depth_coefficient=1.2,
                    include_top=include_top,
                    weights=weights,
                    pooling=pooling,
                    classes=classes,
                    dropout_rate=dropout_rate,
                    drop_connect_rate=drop_connect_rate,
                    data_format=data_format,
                    default_size=256)

if __name__ == "__main__":
    img_input = Input(shape =(256,256, 3))
    # levels = EfficientNetB0(img_input)
    levels = EfficientNetB2(img_input)    
    for i in range(len(levels)):
        print('%s.shape'%levels[i].name, levels[i].shape)