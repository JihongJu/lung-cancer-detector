import time
import pytest
import numpy as np
from preprocessing.volume_image import (
    VolumeImageDataGenerator,
    to_shape)
from preprocessing.image_loader import (
    NPYDataLoader)

import yaml
with open("tests/init_args.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def test_to_shape():
    a = np.ones([3, 3, 3, 1])
    a[:, 1, :, :] = 2
    target_shape = (5, 5, 5, 1)
    b = to_shape(a, target_shape)
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
    datagen = VolumeImageDataGenerator(
        **init_args['volume_image_data_generator']['train']['init'])
    return datagen


@pytest.fixture
def train_vol_loader():
    return NPYDataLoader(**init_args['volume_image_data_loader']['train'])


@pytest.fixture
def test_vol_loader():
    return NPYDataLoader(**init_args['volume_image_data_loader']['val'])


def test_data_loader(train_vol_loader, test_vol_loader):
    assert train_vol_loader.split == "train"
    assert test_vol_loader.split == "val"
    patients1 = train_vol_loader.patients
    patients2 = test_vol_loader.patients
    assert len(np.intersect1d(patients1, patients2)) == 0


def test_data_generator(vol_data_gen, train_vol_loader, test_vol_loader):
    assert vol_data_gen.voxel_mean \
        == init_args['volume_image_data_generator']['train']['init']['voxel_mean']
    assert len(vol_data_gen.image_shape) == 4
    assert len(vol_data_gen.voxel_bounds) == 2
    print('Train')
    train_generator = vol_data_gen.flow_from_loader(
            volume_image_data_loader=train_vol_loader,
            batch_size=32, shuffle=True, seed=42)
    for i in range(16):
	start=time.time()
        batch_x, batch_y = train_generator.next()
        assert batch_x.shape == (32, 96, 96, 96, 1)
	end=time.time()
	print(i, end-start)
    print('Test')
    test_generator = vol_data_gen.flow_from_loader(
        volume_image_data_loader=test_vol_loader,
        batch_size=32, shuffle=True, seed=42)
    for i in range(4):
        batch_x, batch_y = test_generator.next()
        assert batch_x.shape == (32, 96, 96, 96, 1)

