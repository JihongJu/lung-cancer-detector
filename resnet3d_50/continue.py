"""Train script
"""

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
from preprocessing.volume_image import (
    VolumeImageDataGenerator)
from preprocessing.image_loader import (
    NPYDataLoader)
from keras.models import load_model

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
    title = re.sub('^[a-z]+_resnet50_', '', weights.strip('.h5'))

checkpointer = ModelCheckpoint(
    filepath="output/resnet50_checkpoint_{}.h5".format(title),
    verbose=1,
    save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=25)
csv_logger = CSVLogger(
    'output/{}_{}.csv'.format(datetime.datetime.now().isoformat(), title))

train_datagen = VolumeImageDataGenerator(
        **init_args['volume_image_data_generator']['train'])
test_datagen = VolumeImageDataGenerator(
        **init_args['volume_image_data_generator']['test'])

train_vol_loader = NPYDataLoader(
        **init_args['volume_image_data_loader']['train'])
val_vol_loader = NPYDataLoader(
        **init_args['volume_image_data_loader']['val'])

iterator_args = init_args['volume_image_data_generator']['flow_from_loader']
train_iter_args = iterator_args.copy()
train_iter_args['volume_image_data_loader'] = train_vol_loader
val_iter_args = iterator_args.copy()
val_iter_args['volume_image_data_loader'] = val_vol_loader

model = load_model(weights)

model_fit_args = init_args['model']['fit_generator']
model_fit_args['generator'] = train_datagen.flow_from_loader(**train_iter_args)
model_fit_args['validation_data'] = test_datagen.flow_from_loader(
    **val_iter_args)
model_fit_args['callbacks'] = [checkpointer, lr_reducer, early_stopper, csv_logger]

model.fit_generator(**model_fit_args)
model.save('output/resnet50_{}_ctd.h5'.format(title))

