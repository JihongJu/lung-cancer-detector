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
target_size = (224, 224, 224)
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

model = load_model('output/resnet18_stage1.h5')


df_subm = pd.DataFrame(columns=['id', 'cancer'])
for idx, fn in enumerate(test_vol_loader.filenames[:2]):
    x = test_vol_loader.load(fn)
    x = test_datagen.standardize(x)
    print("Predicting {} (Shape: {})".format(fn, x.shape))
    y = model.predict_classes(x, batch_size=1)
    df_subm.loc[idx, 'id'] = fn
    df_subm.loc[idx, 'cancer'] = y

df_subm.to_csv('output/stage1_submission_resnet18',index=False)

