"""Predict script
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import pandas as pd
from keras.utils import np_utils
from preprocessing.volume_image import (
    VolumeDataGenerator,
    NPYDataLoader
)
from keras.models import (load_model)


directory = 'data/data-science-bowl/npy'
image_set = 'stage1'
target_size = (96, 96, 96)
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
    )

model = load_model('output/resnet34_stage1.h5')


df_subm = pd.DataFrame(columns=['id', 'cancer'])
for idx, fn in enumerate(test_vol_loader.filenames):
    x = test_vol_loader.load(fn)
    x = test_datagen.standardize(x)
    x = x[np.newaxis, ...]
    print("Predicting {} (batch shape: {})".format(fn, x.shape))
    proba = model.predict(x, batch_size=1)
    print(proba)
    if proba.shape[-1] > 1:
        y = proba.argmax(axis=-1)
    else:
        y = (proba > 0.5).astype('int32')
    y = y[0][0]
    print("Prediction {fn},{y} ".format(fn=fn, y=y))
    df_subm.loc[idx, 'id'] = fn
    df_subm.loc[idx, 'cancer'] = y

df_subm.to_csv('output/stage1_submission_resnet34.csv',index=False)

