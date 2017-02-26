import pytest
from datasets.data_science_bowl import load_data


def test_data_science_bowl():
    trainval = 'sample_images'

    (X_train, y_train), (X_test, y_test) = load_data(trainval=trainval)
    print(len(X_train), len(y_train))
    assert len(X_train) == len(y_train)
