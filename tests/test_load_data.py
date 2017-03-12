import pytest
import numpy as np
from preprocessing.volume_image import (
    VolumeDataGenerator,
    VolumeDataLoader,
    NPYDataLoader,
    to_shape
)


def test_to_shape():
    a = np.ones([3, 3, 3])
    a[:, 1, :] = 2
    target_size = (5, 5, 5)
    b = to_shape(a, target_size)
    print b
    assert b.shape == (5, 5, 5)
    c = to_shape(a, (5, 3, 5))
    print c
    assert c.shape == (5, 3, 5)
    d = to_shape(a, (5, 2, 5))
    print d
    assert d.shape == (5, 2, 5)


@pytest.fixture
def vol_data_gen():
    datagen = VolumeDataGenerator(
        pixelwise_center=True,
        pixel_mean=0.25,
        pixelwise_normalization=True,
        pixel_bounds=(-1000, 400),
        target_size=(96, 96, 96) 
    )
    return datagen


@pytest.fixture
def train_vol_loader():
    vol_load_args = dict(
        directory='data/data-science-bowl/npy',
        image_set='stage1',
        image_format='npy',
        split='train',
        test_size=0.2,
        random_state=42
    )
    return NPYDataLoader(**vol_load_args)


@pytest.fixture
def test_vol_loader():
    vol_load_args = dict(
        directory='data/data-science-bowl/npy',
        image_set='stage1',
        image_format='npy',
        split='val',
        test_size=0.2,
        random_state=42
        )
    return NPYDataLoader(**vol_load_args)


def test_data_generator(vol_data_gen, train_vol_loader, test_vol_loader):
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


def test_data_loader(train_vol_loader, test_vol_loader):
    filenames1 = train_vol_loader.filenames
    filenames2 = test_vol_loader.filenames
    assert len(np.intersect1d(filenames1, filenames2)) == 0
