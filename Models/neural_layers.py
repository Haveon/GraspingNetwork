import keras.backend as K
from keras.layers import (Conv2D, LeakyReLU, BatchNormalization,
                          Add, Lambda, Dropout)
from keras.regularizers import l1_l2
import tensorflow as tf

def bottleneck_block(input_layer, in_kernels, out_kernels, kernel_size, conv=Conv2D, L1=0, L2=0):
    x = conv(in_kernels, kernel_size=1, padding='same',kernel_regularizer=l1_l2(l1=L1, l2=L2))(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = conv(in_kernels, kernel_size=kernel_size, padding='same',kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = conv(out_kernels, kernel_size=1, padding='same',kernel_regularizer=l1_l2(l1=L1, l2=L2))(x)
    x = BatchNormalization()(x)

    return x

def residual_block(input_layer, bottleneck_kernels, out_kernels, kernel_size, identity=False, conv=Conv2D, L1=0, L2=0):
    if identity:
        b0 = input_layer
    else:
        x  = conv(out_kernels, kernel_size, padding='same', kernel_regularizer=l1_l2(l1=L1, l2=L2))(input_layer)
        b0  = BatchNormalization()(x)

    b1 = bottleneck_block(input_layer, in_kernels=bottleneck_kernels,
                                       out_kernels=out_kernels,
                                       kernel_size=kernel_size,
                                       conv=conv,
                                       L1=L1,
                                       L2=L2)
    x = Add()([b0,b1])
    x = LeakyReLU()(x)

    return x

def bilinear_resize(size):
    return Lambda( lambda x: tf.image.resize_bilinear(x, size, align_corners=True) )

def expand_dims(axis):
    return Lambda( lambda x: K.expand_dims(x, axis=axis) )
