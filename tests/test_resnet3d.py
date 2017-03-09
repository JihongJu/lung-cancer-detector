import pytest
from keras import backend as K
from models.resnet3d import Resnet3DBuilder


def test_resnet3d():
    K.set_image_dim_ordering('tf')
    model = Resnet3DBuilder.build_resnet_18((224, 224, 224, 1), 2)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True, "Failed to build with tensorflow"

    K.set_image_dim_ordering('th')
    model = Resnet3DBuilder.build_resnet_18((1, 224, 224, 224), 2)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    assert True, "Failed to build with theano"
