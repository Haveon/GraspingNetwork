from keras.applications import ResNet50
from keras.models import Model
import keras.backend as K
from keras.layers import (Conv2D, LeakyReLU, Input, BatchNormalization, Add,
                          MaxPooling2D, Lambda, concatenate, Conv3D, Dropout,
                          Activation)
from keras.utils import multi_gpu_model
from keras.optimizers import SGD, Adam
from keras.regularizers import l1_l2
import tensorflow as tf
import numpy as np

class PosNet:
    def __init__(self, input_shape, GPUs=1):
        # self.weighted_loss = np.load('Data/for_training/weighted_loss.npy')
        self.prior_pos = np.load('Data/for_training/prior_pos.npy')
        if GPUs==1:
            self.template_model = self.create_network(input_shape)
            self.model = self.template_model
        elif GPUs>1:
            with tf.device('/cpu:0'):
                self.template_model = self.create_network(input_shape)
                self.model = multi_gpu_model(self.template_model, gpus=GPUs)
        else:
            raise ValueError('GPUs needs to be an integer greater than or equal to 1, not {}'.format(GPUs))

        self.template_model.summary()
        self.model.compile(loss=self.all_way_binary_cross_entropy,
                           optimizer=SGD(lr=LEARNING_RATE, clipnorm=10, momentum=MOMENTUM),
                           metrics=[self.single_accuracy])

    def create_network(self, input_shape):
        KERNEL_NUM = 8
        inp = Input(shape=input_shape)
        x = Conv2D(KERNEL_NUM, 7, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1_REGULARIZER, l2=L2_REGULARIZER))(inp)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(DROPOUT_RATE)(x)

        x = Conv2D(KERNEL_NUM, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1_REGULARIZER, l2=L2_REGULARIZER))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(DROPOUT_RATE)(x)

        # First Residual Block
        x = self.residual_block(x, bottleneck_kernels=KERNEL_NUM,
                                   out_kernels=4*KERNEL_NUM,
                                   kernel_size=3,
                                   identity=False)
        x = Dropout(DROPOUT_RATE)(x)
        for i in range(2):
            x = self.residual_block(x, bottleneck_kernels=KERNEL_NUM,
                                        out_kernels=4*KERNEL_NUM,
                                        kernel_size=3,
                                        identity=True)
            x = Dropout(DROPOUT_RATE)(x)

        # Downsampling
        x = Conv2D(4*KERNEL_NUM, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1_REGULARIZER, l2=L2_REGULARIZER))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(DROPOUT_RATE)(x)

        # Second Residual Block
        x = self.residual_block(x, bottleneck_kernels=2*KERNEL_NUM,
                                   out_kernels=8*KERNEL_NUM,
                                   kernel_size=3,
                                   identity=False)
        x = Dropout(DROPOUT_RATE)(x)
        for i in range(3):
            x = self.residual_block(x, bottleneck_kernels=2*KERNEL_NUM,
                                        out_kernels=8*KERNEL_NUM,
                                        kernel_size=3,
                                        identity=True)
            x = Dropout(DROPOUT_RATE)(x)

        # Downsampling
        x = Conv2D(8*KERNEL_NUM, 3, padding='same', strides=2, kernel_regularizer=l1_l2(l1=L1_REGULARIZER, l2=L2_REGULARIZER))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(DROPOUT_RATE)(x)

        # Third Residual Block
        x = self.residual_block(x, bottleneck_kernels=4*KERNEL_NUM,
                                   out_kernels=16*KERNEL_NUM,
                                   kernel_size=3,
                                   identity=False)
        x = Dropout(DROPOUT_RATE)(x)
        for i in range(5):
            x = self.residual_block(x, bottleneck_kernels=4*KERNEL_NUM,
                                        out_kernels=16*KERNEL_NUM,
                                        kernel_size=3,
                                        identity=True)
            x = Dropout(DROPOUT_RATE)(x)

        # Reshape to (None, 8,8,8, 1024)
        x = self.bilinear_resize((8,8))(x)
        x = self.expand_dims(axis=3)(x)
        repeat = Lambda( lambda x: K.tile( x, (1, 1, 1, 7, 1) ))(x)
        x = concatenate([x,repeat], axis=3)

        # Trainable residual 3D blocks
        x = self.residual_block(x, bottleneck_kernels=8*KERNEL_NUM,
                                    out_kernels=32*KERNEL_NUM,
                                    kernel_size=3,
                                    identity=False,
                                    conv=Conv3D)
        x = Dropout(DROPOUT_RATE)(x)

        for i in range(2): # Originally 2
            x = self.residual_block(x, bottleneck_kernels=8*KERNEL_NUM,
                                       out_kernels=32*KERNEL_NUM,
                                       kernel_size=3,
                                       identity=True,
                                       conv=Conv3D)
        x = Dropout(DROPOUT_RATE)(x)

        x = Conv3D(1, 3, padding='same', activation='linear')(x)
        x = Lambda(lambda x: K.squeeze(x, axis=-1))(x)

        def add_prior(x):
            prior = K.log(K.constant(K.cast_to_floatx(self.prior_pos)))
            prior = self.expand_dims(axis=0)(prior)
            prior = Lambda( lambda prior: K.tile( prior, (K.shape(x)[0], 1, 1, 1) ))(prior)
            return x+prior

        x = Lambda( lambda x: add_prior(x) )(x)
        x = Activation('exponential')(x)
        x = Activation('sigmoid')(x)

        return Model(inp, x)

    def bottleneck_block(self, input_layer, in_kernels, out_kernels, kernel_size, conv=Conv2D):
        x = conv(in_kernels, kernel_size=1, padding='same',kernel_regularizer=l1_l2(l1=L1_REGULARIZER, l2=L2_REGULARIZER))(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = conv(in_kernels, kernel_size=kernel_size, padding='same',kernel_regularizer=l1_l2(l1=L1_REGULARIZER, l2=L2_REGULARIZER))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = conv(out_kernels, kernel_size=1, padding='same',kernel_regularizer=l1_l2(l1=L1_REGULARIZER, l2=L2_REGULARIZER))(x)
        x = BatchNormalization()(x)

        return x

    def residual_block(self, input_layer, bottleneck_kernels, out_kernels, kernel_size, identity=False, conv=Conv2D):
        if identity:
            b0 = input_layer
        else:
            x  = conv(out_kernels, kernel_size, padding='same', kernel_regularizer=l1_l2(l1=L1_REGULARIZER, l2=L2_REGULARIZER))(input_layer)
            b0  = BatchNormalization()(x)

        b1 = self.bottleneck_block(input_layer, in_kernels=bottleneck_kernels,
                                                out_kernels=out_kernels,
                                                kernel_size=kernel_size,
                                                conv=conv)
        x = Add()([b0,b1])
        x = LeakyReLU()(x)

        return x

    def bilinear_resize(self, size):
        return Lambda( lambda x: tf.image.resize_bilinear(x, size, align_corners=True) )

    def expand_dims(self, axis):
        return Lambda( lambda x: K.expand_dims(x, axis=axis) )

    def all_way_binary_cross_entropy(self, y_true, y_pred):
        """
        Both y_true and y_pred have shape (batch_size, 8, 8, 8).

        Returns a Keras tensor that is (batch_size, 1)
        """
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)

        ce_mask = K.cast(K.greater(y_true, 0.5), dtype=K.floatx())
        binary_crossentropy = -(ce_mask*K.log(y_pred+K.epsilon()) + (1-ce_mask)*K.log(1 - y_pred + K.epsilon()))

        filter = K.cast(K.greater(y_true, 0), dtype=K.floatx()) # Binary mask
        loss = filter * binary_crossentropy
        return K.sum(loss, axis=-1)

    def single_accuracy(self, y_true, y_pred):
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)

        filter = K.cast(K.greater(y_true, 0), dtype=K.floatx())

        true_class = K.greater(y_true, 0.5)
        pred_class = K.greater(y_pred, 0.5)
        matches = K.cast(K.equal(true_class, pred_class), dtype=K.floatx())

        matches = K.sum(filter*matches,axis=-1)
        return K.mean(matches)


def data_generator(batch_size, imgs, pos_cubes, neg_cubes):
    total_num = imgs.shape[0]
    half_batch = batch_size//2
    shuffled_indeces = np.arange(total_num)
    while True:
        np.random.shuffle(shuffled_indeces)
        for i in range(total_num//half_batch):
            current_indeces = shuffled_indeces[i*half_batch:(i+1)*half_batch]
            current_images = imgs[current_indeces]
            # TODO: Preprocess
            batch_images = np.concatenate([current_images, current_images], axis=0)

            batch_cubes = np.zeros([2*current_indeces.shape[0],8,8,8])
            for j,cur_ind in enumerate(current_indeces):
                pz,py,px = pos_cubes[cur_ind]
                nz,ny,nx = neg_cubes[cur_ind]
                batch_cubes[j, pz,py,px] = 1
                batch_cubes[half_batch+j, nz,ny,nx] = 0.2

            yield batch_images, batch_cubes

if __name__ == '__main__':
    import numpy as np
    from keras.applications import resnet50
    from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

    # Hyperparameters
    BATCH_SIZE = 64
    L1_REGULARIZER = 0#0.001
    L2_REGULARIZER = 1e-4#0.001
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    DROPOUT_RATE = 0.3

    # Callbacks
    csv_logger = CSVLogger('training.log')
    model_checkpoint = ModelCheckpoint('checkpoints/best_model.h5', save_best_only=True)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_delta=1e-4)

    print('Loading Images')
    #train_images = np.zeros([31062,224,224,3], dtype=np.int8)
    train_images = np.load('Data/for_training/training_image_data.npy')
    val_images = np.load('Data/for_training/validation_image_data.npy')
    val_images = np.concatenate([val_images,val_images],axis=0)

    print('Preprocessing Images')
    train_images = resnet50.preprocess_input(train_images)
    val_images = resnet50.preprocess_input(val_images)

    print('Loading Targets')
    train_pos_cubes = np.load('Data/for_training/training_positive_cubes.npy').astype(np.int32)[:, 1:]
    train_neg_cubes = np.load('Data/for_training/training_negative_cubes.npy').astype(np.int32)[:, 1:]

    val_pos_cubes = np.load('Data/for_training/validation_positive_cubes.npy').astype(np.int32)[:, 1:]
    val_neg_cubes = np.load('Data/for_training/validation_negative_cubes.npy').astype(np.int32)[:, 1:]

    validation_cubes = np.zeros([2000, 8,8,8])
    for i in range(1000):
        pz,py,px = val_pos_cubes[i]
        validation_cubes[i, pz,py,px] = 1
        nz,ny,nx = val_neg_cubes[i]
        validation_cubes[i+1000, nz,ny,nx] = 0.2

    print('Building Network')
    network = PosNet([224,224,3])
    print('Configuring Generator')
    datagen = data_generator(BATCH_SIZE, train_images, train_pos_cubes, train_neg_cubes)
    print('Beginning Training')
    batch_per_epoch = (2*train_images.shape[0])/BATCH_SIZE
    network.model.fit_generator(datagen, steps_per_epoch=batch_per_epoch,
                                         epochs=100,
                                         validation_data=(val_images,validation_cubes),
                                         callbacks=[csv_logger, model_checkpoint, reduce_lr_on_plateau])
