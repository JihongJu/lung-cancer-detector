"""Train script."""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import re
import datetime
import argparse
import numpy as np
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint)
from keras.optimizers import Adam
from preprocessing.volume_image import (
    VolumeImageDataGenerator)
from preprocessing.image_loader import (
    NPYDataLoader)
from resnet3d import Resnet3DBuilder

import yaml
with open("init_args.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

parser = argparse.ArgumentParser(description='Continue a training')
parser.add_argument('weights', help='Trained weights')
args = parser.parse_args()
if args.weights:
    weights = args.weights
    title = re.sub('^output/resnet18(_checkpoint)*_', '', weights.strip('.h5'))

nb_classes = init_args['volume_image_data_generator']['train'][
    'flow_from_loader']['nb_classes']

print("Continue training {}.".format(title))
checkpointer = ModelCheckpoint(
    filepath="output/resnet18_checkpoint_{}_ctd.h5".format(title),
    verbose=1,
    save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10, min_lr=1e-6)
early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=100)
csv_logger = CSVLogger(
    'output/{}_{}_ctd.csv'.format(datetime.datetime.now().isoformat(), title))

train_datagen = VolumeImageDataGenerator(
        **init_args['volume_image_data_generator']['train']['init'])
val_datagen = VolumeImageDataGenerator(
        **init_args['volume_image_data_generator']['val']['init'])

train_vol_loader = NPYDataLoader(
        **init_args['volume_image_data_loader']['train'])
val_vol_loader = NPYDataLoader(
        **init_args['volume_image_data_loader']['val'])

train_iter_args = init_args['volume_image_data_generator']['train']['flow_from_loader']
train_iter_args['volume_image_data_loader'] = train_vol_loader
val_iter_args = init_args['volume_image_data_generator']['val']['flow_from_loader']
val_iter_args['volume_image_data_loader'] = val_vol_loader

image_shape = train_datagen.image_shape
regularization_factor = 1
model = Resnet3DBuilder.build_resnet_18(image_shape, nb_classes, regularization_factor)
model.load_weights(weights)
compile_args = init_args['model']['compile']
compile_args['optimizer'] = Adam(lr=1e-4)
model.compile(**compile_args)

model_fit_args = init_args['model']['fit_generator']
model_fit_args['generator'] = train_datagen.flow_from_loader(**train_iter_args)
model_fit_args['validation_data'] = val_datagen.flow_from_loader(
    **val_iter_args)
model_fit_args['callbacks'] = [checkpointer, lr_reducer, early_stopper, csv_logger]

model.fit_generator(**model_fit_args)
model.save('output/resnet18_{}_ctd.h5'.format(title))

