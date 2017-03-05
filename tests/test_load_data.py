import pytest
from preprocessing.volume_image import (
    VolumeDataGenerator,
    VolumeDataLoader
)


@pytest.fixture
def vol_data_gen():
    datagen = VolumeDataGenerator()
    return datagen


@pytest.fixture
def train_vol_loader():
    vol_load_args = dict(
        directory='data/dcm',
        image_set='sample_images',
        image_format='dcm',
        split='train',
        test_size=0.2,
        random_state=42,
        target_size=(512, 512, 128)
    )
    return VolumeDataLoader(**vol_load_args)


@pytest.fixture
def test_vol_loader():
    vol_load_args = dict(
        directory='data/dcm',
        image_set='sample_images',
        image_format='dcm',
        split='val',
        test_size=0.2,
        random_state=42,
        target_size=(512, 512, 128)
        )
    return VolumeDataLoader(**vol_load_args)


def test_data_generator(vol_data_gen, train_vol_loader, test_vol_loader):
    print('Train')
    train_generator = vol_data_gen.flow_from_loader(
            volume_data_loader=train_vol_loader,
            batch_size=1, shuffle=True, seed=42)
    for i in range(16):
        batch_x, batch_y = train_generator.next()
        assert batch_x.shape == (1, 512, 512, 128, 1)
    print('Test')
    test_generator = vol_data_gen.flow_from_loader(
        volume_data_loader=test_vol_loader,
        batch_size=1, shuffle=True, seed=42)
    for i in range(4):
        batch_x, batch_y = test_generator.next()
        assert batch_x.shape == (1, 512, 512, 128, 1)
        
