"""Train script
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from keras.utils import np_utils
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint)
from preprocessing.volume_image import (
    VolumeDataGenerator,
    NPYDataLoader
)
from models.resnet3d import Resnet3DBuilder

directory = 'data/data-science-bowl/npy'
image_set = 'stage1'
target_size = (96, 96, 96)
test_size = 0.2
random_state = 42
batch_size = 32
class_mode = 'binary'
nb_classes = 1
#samples_per_epoch = 16
#nb_val_samples = 4
samples_per_epoch = 1116
nb_val_samples = 280
nb_epoch = 100
data_augmentation = True


checkpointer = ModelCheckpoint(filepath="/tmp/resnet34_weights.hdf5", verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100)
csv_logger = CSVLogger('output/resnet34_{}.csv'.format(image_set))

train_datagen = VolumeDataGenerator(
    pixelwise_center=True,
    pixel_mean=0.25,
    pixelwise_normalization=True,
    pixel_bounds=(-1000, 400),
    imlearn_resampler=None,
    #imlearn_resampler='rus',
    target_size=target_size
)
test_datagen = VolumeDataGenerator(
    pixelwise_center=True,
    pixel_mean=0.25,
    pixelwise_normalization=True,
    pixel_bounds=(-1000, 400),
    imlearn_resampler=None,
    target_size=target_size
)
train_vol_loader = NPYDataLoader(
    directory=directory,
    image_set=image_set,
    image_format='npy',
    split='train',
    test_size=test_size,
    random_state=random_state
)
test_vol_loader = NPYDataLoader(
    directory=directory,
    image_set=image_set,
    image_format='npy',
    split='val',
    test_size=test_size,
    random_state=random_state
    )

model = Resnet3DBuilder.build_resnet_34(target_size + (1,), nb_classes)
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])


model.fit_generator(
    train_datagen.flow_from_loader(
        volume_data_loader=train_vol_loader,
        class_mode=class_mode,
        nb_classes=nb_classes,
        batch_size=batch_size,
        shuffle=True),
    samples_per_epoch=samples_per_epoch,
    nb_epoch=nb_epoch,
    validation_data=test_datagen.flow_from_loader(
        volume_data_loader=test_vol_loader,
        batch_size=batch_size,
        class_mode=class_mode,
        nb_classes=nb_classes,
        shuffle=True),
    nb_val_samples=nb_val_samples,
    verbose=1, max_q_size=100,
    callbacks=[lr_reducer, early_stopper, csv_logger]
)
model.save('output/resnet34_{}.h5'.format(image_set))
