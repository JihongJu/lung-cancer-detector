"""Train script
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
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

import yaml
with open("config.yml", 'r') as stream:
    try:
        config_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

image_set = config_args['volume_data_loader']['train']['image_set']
target_size = tuple(config_args['volume_data_generator']['train']['target_size'])
nb_classes = config_args['volume_data_generator']['flow_from_loader']['nb_classes']


checkpointer = ModelCheckpoint(filepath="/tmp/resnet34_weights.hdf5", verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100)
csv_logger = CSVLogger('output/resnet34_{}.csv'.format(image_set))

train_datagen = VolumeDataGenerator(
        **config_args['volume_data_generator']['train'])
test_datagen = VolumeDataGenerator(
        **config_args['volume_data_generator']['test'])

train_vol_loader = NPYDataLoader(
        **config_args['volume_data_loader']['train'])
val_vol_loader = NPYDataLoader(
        **config_args['volume_data_loader']['val'])

iterator_args = config_args['volume_data_generator']['flow_from_loader']
train_iter_args = iterator_args.copy()
train_iter_args['volume_data_loader'] = train_vol_loader
val_iter_args = iterator_args.copy()
val_iter_args['volume_data_loader'] = val_vol_loader

model = Resnet3DBuilder.build_resnet_34(target_size + (1,), nb_classes)
model.compile(**config_args['model']['compile'])

model_fit_args = config_args['model']['fit_generator']
model_fit_args['generator']=train_datagen.flow_from_loader(**train_iter_args)
model_fit_args['validation_data']=test_datagen.flow_from_loader(**val_iter_args)
model_fit_args['callbacks']=[lr_reducer, early_stopper, csv_logger]

model.fit_generator(**model_fit_args)
model.save('output/resnet34_{}.h5'.format(image_set))
