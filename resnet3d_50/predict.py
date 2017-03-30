"""test script
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import re
import argparse
import pandas as pd
from keras.models import load_model
from preprocessing.volume_image import (
    VolumeImageDataGenerator)
from preprocessing.image_loader import (
    NPYDataLoader)

import yaml
with open("init_args.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

parser = argparse.ArgumentParser(description='Predict given weights')
parser.add_argument('weights', help='Trained weights')
args = parser.parse_args()
if args.weights:
    weights = args.weights
    title = re.sub('^output/resnet50_[a-z]+_', '', weights.strip('.h5'))
    print("Prediction by {}.".format(title))

test_datagen = VolumeImageDataGenerator(
        **init_args['volume_image_data_generator']['test']['init'])

test_vol_loader = NPYDataLoader(
        **init_args['volume_image_data_loader']['test'])

test_iter_args = init_args['volume_image_data_generator']['test']['flow_from_loader']
test_iter_args['volume_image_data_loader'] = test_vol_loader


model = load_model(weights)

model_pred_args = init_args['model']['predict_generator']
model_pred_args['generator'] = test_datagen.flow_from_loader(
        **test_iter_args)
predictions = model.predict_generator(**model_pred_args)
print(predictions.flatten())
print(len(predictions))

df_subm = pd.DataFrame(columns=['id', 'cancer'])
df_subm['id'] = test_vol_loader.patients
df_subm['cancer'] = predictions.flatten()

df_subm.to_csv('output/stage1_submission_resnet50_{}.csv'.format(title), index=False)
