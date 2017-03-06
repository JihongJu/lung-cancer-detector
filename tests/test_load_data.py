import pytest
import numpy as np
from preprocessing.volume_image import (
    VolumeDataGenerator,
    VolumeDataLoader,
    NPYDataLoader,
    pad_to_shape
)


def test_pad_to_shape():
    a = np.ones([3, 3, 3])
    target_size = (5, 5, 5)
    b = pad_to_shape(a, target_size)
    print b
    assert b.shape == (5, 5, 5)
    c = pad_to_shape(a, (5, 3, 5))
    print c
    assert c.shape == (5, 3, 5)


@pytest.fixture
def vol_data_gen():
    datagen = VolumeDataGenerator(
        pixelwise_center=True,
        pixel_mean=0.25,
        pixelwise_normalization=True,
        pixel_bounds=(-1000, 400),
    )
    return datagen


@pytest.fixture
def train_vol_loader():
    vol_load_args = dict(
        directory='data/data-science-bowl/npy',
        image_set='sample_images',
        image_format='npy',
        split='train',
        test_size=0.2,
        random_state=42,
        target_size=(448, 448, 448)
    )
    return NPYDataLoader(**vol_load_args)


@pytest.fixture
def test_vol_loader():
    vol_load_args = dict(
        directory='data/data-science-bowl/npy',
        image_set='sample_images',
        image_format='npy',
        split='val',
        test_size=0.2,
        random_state=42,
        target_size=(448, 448, 448)
        )
    return NPYDataLoader(**vol_load_args)


def test_data_generator(vol_data_gen, train_vol_loader, test_vol_loader):
    print('Train')
    train_generator = vol_data_gen.flow_from_loader(
            volume_data_loader=train_vol_loader,
            batch_size=1, shuffle=True, seed=42)
    for i in range(16):
        batch_x, batch_y = train_generator.next()
        assert batch_x.shape == (1, 448, 448, 448, 1)
    print('Test')
    test_generator = vol_data_gen.flow_from_loader(
        volume_data_loader=test_vol_loader,
        batch_size=1, shuffle=True, seed=42)
    for i in range(4):
        batch_x, batch_y = test_generator.next()
        assert batch_x.shape == (1, 448, 448, 448, 1)
