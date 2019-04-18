from keras.utils import multi_gpu_model
from keras.applications import resnet50
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import numpy as np

def prepare_model(create_network, input_shape, loss, optimizer, metrics, prior, GPUs=1, L1=0, L2=0, dropout=0):
    if GPUs==1:
        template_model = create_network(input_shape, prior, L1=L1, L2=L2, dropout=dropout)
        model = template_model
    elif GPUs>1:
        with tf.device('/cpu:0'):
            template_model = create_network(input_shape, prior, L1=L1, L2=L2, dropout=dropout)
            model = multi_gpu_model(template_model, gpus=GPUs)
    else:
        raise ValueError('GPUs needs to be an integer greater than or equal to 1, not {}'.format(GPUs))

    template_model.summary()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return (template_model, model)

def data_generator(batch_size, imgs, pos_cubes, neg_cubes):
    total_num = imgs.shape[0]
    half_batch = batch_size//2
    shuffled_indeces = np.arange(total_num)
    while True:
        np.random.shuffle(shuffled_indeces)
        for i in range(total_num//half_batch):
            current_indeces = shuffled_indeces[i*half_batch:(i+1)*half_batch]
            current_images = imgs[current_indeces]

            batch_images = np.concatenate([current_images, current_images], axis=0)

            batch_cubes = np.zeros([2*current_indeces.shape[0],8,8,8])
            for j,cur_ind in enumerate(current_indeces):
                pz,py,px = pos_cubes[cur_ind]
                nz,ny,nx = neg_cubes[cur_ind]
                batch_cubes[j, pz,py,px] = 1
                batch_cubes[half_batch+j, nz,ny,nx] = 0.2

            yield batch_images, batch_cubes

def load_data():
    print('\tLoading Images')
    #train_images = np.zeros([31062,224,224,3], dtype=np.int8)
    train_images = np.load('Data/for_training/small_image_set.npy')
    val_images = np.load('Data/for_training/validation_image_data.npy')[571:]
    val_images = np.concatenate([val_images,val_images],axis=0)

    print('\tPreprocessing Images')
    train_images = resnet50.preprocess_input(train_images)
    val_images = resnet50.preprocess_input(val_images)

    print('\tLoading Targets')
    train_pos_cubes = np.load('Data/for_training/training_positive_cubes.npy').astype(np.int32)[:512, 1:]
    train_neg_cubes = np.load('Data/for_training/training_negative_cubes.npy').astype(np.int32)[:512, 1:]

    val_pos_cubes = np.load('Data/for_training/validation_positive_cubes.npy').astype(np.int32)[571:, 1:]
    val_neg_cubes = np.load('Data/for_training/validation_negative_cubes.npy').astype(np.int32)[571:, 1:]

    validation_cubes = np.zeros([(1000-571)*2, 8,8,8])
    for i in range(1000-571):
        pz,py,px = val_pos_cubes[i]
        validation_cubes[i, pz,py,px] = 1
        nz,ny,nx = val_neg_cubes[i]
        validation_cubes[i+(1000-571), nz,ny,nx] = 0.2

    return (train_images, val_images), (train_pos_cubes, train_neg_cubes), validation_cubes

if __name__ == '__main__':
    from Models.resnet_style import create_network

    # Hyperparameters
    BATCH_SIZE = 64
    L1_REGULARIZER = 0
    L2_REGULARIZER = 0
    LEARNING_RATE = 1e-5
    MOMENTUM = 0
    DROPOUT_RATE = 0

    # Callbacks
    csv_logger = CSVLogger('training.log')
    model_checkpoint = ModelCheckpoint('checkpoints/best_model.h5', save_best_only=True)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=1e-4)

    print('Building Network')
    from keras.optimizers import SGD, Adam
    from Models.loss_and_metrics import single_accuracy, all_way_binary_cross_entropy
    opt = Adam(lr=LEARNING_RATE, clipnorm=1)

    (template, model) = prepare_model(create_network, input_shape=[224,224,3],
                                                      loss=all_way_binary_cross_entropy,
                                                      optimizer=opt,
                                                      metrics=[single_accuracy],
                                                      GPUs=1,
                                                      L1=L1_REGULARIZER,
                                                      L2=L2_REGULARIZER,
                                                      dropout=DROPOUT_RATE,
                                                      prior=np.load('Data/for_training/prior_pos.npy'))

    print('Loading Data')
    (train_images, val_images), (train_pos_cubes, train_neg_cubes), validation_cubes = load_data()

    print('Configuring Generator')
    datagen = data_generator(BATCH_SIZE, train_images, train_pos_cubes, train_neg_cubes)

    print('Beginning Training')
    batch_per_epoch = (2*train_images.shape[0])/BATCH_SIZE
    model.fit_generator(datagen, steps_per_epoch=batch_per_epoch,
                                 epochs=1000,
                                 validation_data=(val_images,validation_cubes),
                                 callbacks=[csv_logger, model_checkpoint])#, reduce_lr_on_plateau])
