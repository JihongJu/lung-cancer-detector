from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os
import dicom
import numpy as np
import pandas as pd
import nipy as ni
import nibabel as nib
import keras.backend as K
from keras.preprocessing.image import Iterator
from nipy.core.api import Image, vox2mni
from nipy.labs.datasets.volumes.volume_img import VolumeImg
from sklearn.model_selection import train_test_split


class VolumeDataGenerator(object):

    def __init__(self, preprocessing_function=None, dim_ordering='default'):
        # image_dim_ordering
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering should be "tf" '
                             'with (dim1, dim2, dim3, channel) or '
                             '"th" (channel, dim1, dim2, dim3). '
                             'Received arg: ', dim_ordering)
        if dim_ordering == 'th':
            self.channel_axis = 1
            self.dim1_axis = 2
            self.dim2_axis = 3
            self.dim3_axis = 4
        if dim_ordering == 'tf':
            self.channel_axis = 4
            self.dim1_axis = 1
            self.dim2_axis = 2
            self.dim3_axis = 3
        self.dim_ordering = dim_ordering


    def flow_from_loader(self, volume_data_loader,
                         batch_size=1, shuffle=True, seed=None):
        return VolumeLoaderIterator(
            volume_data_loader, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed
        )

    def standardize(self, x):
        """standardize inputs including featurewise/samplewise
        center/normalization, etc."""
        # if self.preprocessing_function:
        #     x = self.preprocessing_function(x)
        pass

    def random_transform(self, x):
        """random transforms including flipping"""
        pass


class VolumeLoaderIterator(Iterator):

    def __init__(self, volume_data_loader, volume_data_generator,
                 batch_size=1, shuffle=False, seed=None):
        self.volume_data_loader = volume_data_loader
        self.volume_data_generator = volume_data_generator

        self.filenames = volume_data_loader.filenames
        self.nb_sample = len(self.filenames)
        self.image_shape = volume_data_loader.image_shape
        self.classes = volume_data_loader.classes
        # compute mean
        # mean and rescale (max-min) for normalization
        self.mean = None
        self.rescale = 2192.

        super(VolumeLoaderIterator, self).__init__(self.nb_sample, batch_size,
                                                   shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        batch_x = np.zeros((current_batch_size,) + self.image_shape,
                           dtype=K.floatx())
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            print("Loading {}, {}".format(j, fname))
            img = self.volume_data_loader.load(fname)
            # augmentation goes here
            x = img_to_array(img)
            x -= self.mean
            batch_x[i] = x
        batch_y = self.classes[index_array]

        return batch_x, batch_y


class VolumeDataLoader(object):
    """ Helper class for loading images
    # Data structure Overview
    \root
        \data
            \dcm
                \sample_images
                sample_images_labels.csv
                \stage1
                stage1_labels.csv
            \nii
                \sample_images
                sample_images_labels.csv
                \stage1
                stage1_labels.csv
    """
    def __init__(self, directory, image_set, image_format='dcm',
                 split='train', test_size=0.2, random_state=42,
                 target_size=(512, 512, 128), dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='dcm'):
        self.directory = directory
        self.image_set = image_set
        self.image_dir = os.path.join(directory, image_set)
        if not os.path.exists(self.image_dir):
            raise IOError('Directory {} does not exist. Please provide a '
                          'valid directory.'.format(self.image_dir))

        if split not in {'train', 'val', 'trainval'}:
            raise ValueError('dataset split must be in '
                             '{"train", "val", "trainval"} '
                             'Got {}'.format(split))
        self.split = split
        self.test_size = test_size

        try:
            df_labels = pd.read_csv(
                os.path.join(self.directory,
                             "{}_labels.csv".format(self.image_set)))
        except IOError:
            raise
        self.filenames = df_labels['id'].values
        self.classes = df_labels['cancer'].values.astype('int')

        self.image_format = image_format
        white_list_formats = {'dcm', 'nii'}
        if image_format not in white_list_formats:
            raise ValueError('Invalid image format:', image_format,
                             '; expected "dcm", "nii".')

        self.target_size = tuple(target_size)
        if len(target_size) != 3:
            raise ValueError('Volumetric data requires 3 dimensions for '
                             'target_size. Got {} dimensions'.format(
                                 len(target_size)))

        if dim_ordering not in {'default', 'tf', 'th'}:
            raise ValueError('Invalid dim ordering:', dim_ordering,
                             '; expected "default", "tf" or "th".')
        self.dim_ordering = dim_ordering
        if dim_ordering == 'default':
            self.dim_ordering = K.image_dim_ordering()
        if self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size

        # split train test or keep all
        filenames_train, filenames_test, classes_train, classes_test \
            = train_test_split(self.filenames, self.classes,
                               test_size=test_size, random_state=random_state)
        if self.split == 'train':
            self.filenames = filenames_train
            self.classes = classes_train
        elif self.split == 'val':
            self.filenames = filenames_test
            self.classes = classes_test

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

    def load(self, fn):
        """Image series load method.
        # Arguments
            fn: filename of the image (without extension suffix)
        # Returns
            arr: numpy array of shape self.target_size
        """
        # load image
        if self.image_format == 'nii':
            img_path = os.path.join(self.image_dir,
                                    'Axial_{}.nii.gz'.format(fn))
            if not os.path.exists(img_path):
                raise IOError('Image {} does not exist.'.format(img_path))
            img = ni.load_image(img_path)
        elif self.image_format == 'dcm':
            img_path = os.path.join(self.image_dir, fn)
            if not os.path.exists(img_path):
                raise IOError('Image {} does not exist.'.format(img_path))
            img_filenames = os.listdir(img_path)
            nb_slices = len(img_filenames)
            img = np.zeros((512, 512, nb_slices), dtype='int16')
            for f_id, filename in enumerate(img_filenames):
                dc = dicom.read_file(os.path.join(img_path, filename))
                img[..., f_id] = dc.pixel_array
        # resample to fix size
        vol_img = VolumeImg(data=img, affine=np.eye(4),
                            world_space='')
        resampled_img = vol_img.as_volume_img(affine=np.eye(4),
                                              shape=self.target_size)

        return resampled_img

    def save(self, arr):
        pass


def img_to_array(img, dim_ordering='default'):
    """Converts VolumeImg to arr
        # Arguments
        img: VolumeImg instance.
        dim_ordering: Image data format.
    # Returns
        A 4D Numpy array.
    """
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Unknown dim_ordering: ', dim_ordering)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = img.get_data().astype(K.floatx())
    if len(x.shape) == 4:
        if dim_ordering == 'th':
            x = x.transpose(3, 0, 1, 2)
    elif len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x
