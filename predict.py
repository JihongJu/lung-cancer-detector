"""Predict script
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
target_size = (224, 224, 224)
batch_size = 1
class_mode = 'binary'
nb_classes = 1
samples_per_epoch = 1116
nb_val_samples = 280
nb_epoch = 20
data_augmentation = True


test_datagen = VolumeDataGenerator(
    pixelwise_center=True,
    pixel_mean=0.25,
    pixelwise_normalization=True,
    pixel_bounds=(-1000, 400),
    target_size=target_size
)

test_vol_loader = NPYDataLoader(
    directory=directory,
    image_set=image_set,
    image_format='npy',
    split='test',
    random_state=random_state
    )

model = load_model('output/resnet18_stage1.h5')

#model.predict_classes(
#    test_datagen.flow_from_loader(
#        volume_data_loader=test_vol_loader,
#        class_mode=class_mode,
#        nb_classes=nb_classes,
#        batch_size=batch_size,
#        shuffle=False,)
#model.save('output/resnet18_{}.h5'.format(image_set))
