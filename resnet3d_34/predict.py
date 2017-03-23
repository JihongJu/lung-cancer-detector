"""test script
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import pandas as pd
from keras.models import (load_model)

from preprocessing.volume_image import (
    VolumeImageDataGenerator)
from preprocessing.image_loader import (
    NPYDataLoader)


import yaml
with open("config.yml", 'r') as stream:
    try:
        config_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

test_datagen = VolumeImageDataGenerator(
        **config_args['volume_image_data_generator']['test'])

test_vol_loader = NPYDataLoader(
        **config_args['volume_image_data_loader']['test'])

iterator_args = config_args['volume_image_data_generator']['flow_from_loader']
test_iter_args = iterator_args.copy()
test_iter_args['volume_image_data_loader'] = test_vol_loader


model = load_model('output/resnet34_stage1.h5')


df_subm = pd.DataFrame(columns=['id', 'cancer'])
for idx, fn in enumerate(test_vol_loader.filenames):
    x = test_vol_loader.load(fn)
    x = test_datagen.standardize(x)
    x = x[np.newaxis, ...]
    print("Predicting {} (batch shape: {})".format(fn, x.shape))
    proba = model.test(x, batch_size=1)
    print(proba)
    if proba.shape[-1] > 1:
        y = proba.argmax(axis=-1)
    else:
        y = (proba > 0.5).astype('int32')
    y = y[0][0]
    print("Prediction: {fn},{y} ".format(fn=fn, y=y))
    df_subm.loc[idx, 'id'] = fn
    df_subm.loc[idx, 'cancer'] = y

df_subm.to_csv('output/stage1_submission_resnet34.csv',index=False)
