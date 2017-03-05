"""Train script
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from .preprocessing.volume_image import (
    VolumeDataGenerator,
    VolumeDataLoader
)
from .models.resnet3d import Resnet3DBuilder


lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_stage1.csv')

target_size = (512, 512, 128)
test_size = 0.2
random_state = 42
batch_size = 1
nb_classes = 2
nb_epoch = 200
data_augmentation = True

train_datagen = VolumeDataGenerator()
test_datagen = VolumeDataGenerator()
train_vol_loader = VolumeDataLoader(
    directory='data/dcm',
    image_set='sample_images',
    image_format='dcm',
    split='train',
    test_size=test_size,
    random_state=random_state,
    target_size=target_size
)
test_vol_loader = VolumeDataLoader(
    directory='data/dcm',
    image_set='sample_images',
    image_format='dcm',
    split='val',
    test_size=test_size,
    random_state=random_state,
    target_size=(512, 512, 128)
    )

model = Resnet3DBuilder.build_resnet_18((target_size,) + (1,), 2)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
model.fit_generator(
    train_datagen.flow_from_loader(
        volume_data_loader=train_vol_loader, batch_size=batch_size,
        shuffle=True),
    nb_epoch=nb_epoch,
    validation_generator=test_datagen.flow_from_loader(
        volume_data_loader=train_vol_loader, batch_size=batch_size,
        shuffle=True),
    callbacks=[lr_reducer, early_stopper, csv_logger]
)
