import pytest
import numpy as np
from preprocessing.volume_image import (
    VolumeDataGenerator,
    NPYDataLoader,
    to_shape
)

import yaml
with open("config.yml", 'r') as stream:
    try:
        config_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def test_to_shape():
    a = np.ones([3, 3, 3, 1])
    a[:, 1, :, :] = 2
    target_size = (5, 5, 5, 1)
    b = to_shape(a, target_size)
    print b
    assert b.shape == (5, 5, 5, 1)
    c = to_shape(a, (5, 3, 5, 1))
    print c
    assert c.shape == (5, 3, 5, 1)
    d = to_shape(a, (5, 2, 5, 1))
    print d
    assert d.shape == (5, 2, 5, 1)


@pytest.fixture
def vol_data_gen():
    datagen = VolumeDataGenerator(**config_args['volume_data_generator']['train'])
    return datagen


@pytest.fixture
def train_vol_loader():
    return NPYDataLoader(**config_args['volume_data_loader']['train'])


@pytest.fixture
def test_vol_loader():
    return NPYDataLoader(**config_args['volume_data_loader']['val'])


def test_data_loader(train_vol_loader, test_vol_loader):
    assert train_vol_loader.split == "train"
    assert test_vol_loader.split == "val"
    filenames1 = train_vol_loader.filenames
    assert len(filenames1) == config_args['model']['fit_generator']['samples_per_epoch']
    filenames2 = test_vol_loader.filenames
    assert len(filenames2) == config_args['model']['fit_generator']['nb_val_samples']
    assert len(np.intersect1d(filenames1, filenames2)) == 0

def test_data_generator(vol_data_gen, train_vol_loader, test_vol_loader):
    assert vol_data_gen.pixel_mean == config_args['volume_data_generator']['train']['pixel_mean']
    assert len(vol_data_gen.target_size) == 3
    assert len(vol_data_gen.pixel_bounds) == 2
    print('Train')
    train_generator = vol_data_gen.flow_from_loader(
            volume_data_loader=train_vol_loader,
            batch_size=32, shuffle=True, seed=42)
    for i in range(16):
        batch_x, batch_y = train_generator.next()
        assert batch_x.shape == (32, 96, 96, 96, 1)
    print('Test')
    test_generator = vol_data_gen.flow_from_loader(
        volume_data_loader=test_vol_loader,
        batch_size=32, shuffle=True, seed=42)
    for i in range(4):
        batch_x, batch_y = test_generator.next()
        assert batch_x.shape == (32, 96, 96, 96, 1)

