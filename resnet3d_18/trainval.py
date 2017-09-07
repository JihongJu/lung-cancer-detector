"""Train script
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import datetime
import requests
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

# generate a random training title
r = requests.get('https://frightanic.com/goodies_content/docker-names.php')
if r.raise_for_status():
    raise
title = r.text.rstrip()

# parset a training title
parser = argparse.ArgumentParser(description='Continue a training.')
parser.add_argument('-t', help='The title of the training to continue')
args = parser.parse_args()
if args.t:
    title = args.t

nb_classes = init_args['volume_image_data_generator']['train'][
    'flow_from_loader']['nb_classes']


checkpointer = ModelCheckpoint(
    filepath="output/resnet18_checkpoint_{}.h5".format(title),
    verbose=1,
    save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10, min_lr=1e-6)
early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=50)
csv_logger = CSVLogger(
    'output/{}_{}.csv'.format(datetime.datetime.now().isoformat(), title))

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
regularization_factor = 5
model = Resnet3DBuilder.build_resnet_18(image_shape, nb_classes, regularization_factor)
compile_args = init_args['model']['compile']
compile_args['optimizer'] = Adam(lr=1e-3)
model.compile(**compile_args)

model_fit_args = init_args['model']['fit_generator']
model_fit_args['generator'] = train_datagen.flow_from_loader(**train_iter_args)
model_fit_args['validation_data'] = val_datagen.flow_from_loader(
    **val_iter_args)
model_fit_args['callbacks'] = [checkpointer, lr_reducer, early_stopper, csv_logger]

model.fit_generator(**model_fit_args)
model.save('output/resnet18_{}.h5'.format(title))
