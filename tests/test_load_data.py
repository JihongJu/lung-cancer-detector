import pytest
from ..datasets.data_science_bowl import load_data


def test_data_science_bowl():
    trainval = 'sample_images'

    x, y = load_data(trainval=trainval, dimension=3)
    print(len(x), len(y))
    assert len(x) == len(y)
