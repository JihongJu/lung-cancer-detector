from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os
import dicom
import scipy
import numpy as np
import pandas as pd
import nipy as ni
import nibabel as nib
import keras.backend as K
from keras.preprocessing.image import Iterator
from nipy.core.api import Image, vox2mni
from nipy.labs.datasets.volumes.volume_img import VolumeImg
from sklearn.model_selection import train_test_split


def pixelwise_normalize(image, pixel_bounds):
    if len(pixel_bounds) != 2:
        raise ValueError("Expected tuple of pixel bounds "
                         "(min_bound, max_bound). Got {}".format(pixel_bounds))
    min_bound = pixel_bounds[0]
    max_bound = pixel_bounds[1]
    image = (image - min_bound) / (max_bound - min_bound)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


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
    if isinstance(img, np.ndarray):
        x = img.astype(K.floatx())
    else:
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


def pad_to_shape(arr, target_size):
    """Pad arrays such that it has the shape of target_size
    """
    arr_shape = arr.shape
    npad = ()
    for dim in range(len(arr_shape)):
        diff = target_size[dim] - arr_shape[dim]
        before = int(diff / 2)
        after = diff - before
        npad += ((before, after),)
    pad_arr = np.pad(arr, pad_width=npad,
                     mode='constant',
                     constant_values=0)
    return pad_arr


class VolumeDataGenerator(object):

    def __init__(self,
                 pixelwise_center=None,
                 pixel_mean=None,
                 pixelwise_normalization=None,
                 pixel_bounds=None,
                 preprocessing_function=None, dim_ordering='default'):
        """
        # Arguments
            pixelwise_center: substract pixel mean
            pixel_mean: dataset_mean (e.g. 0.25 for LUNA2016)
            pixelwise_normalization: scaled by pixel_bounds
            pixel_bounds: tuple of pixel bounds (min_bound, max_bound)
        """
        self.pixelwise_center = pixelwise_center
        self.pixel_mean = pixel_mean
        self.pixelwise_normalization = pixelwise_normalization
        self.pixel_bounds = pixel_bounds
        self.preprocessing_function = preprocessing_function
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
                         class_mode='binary',
                         batch_size=1, shuffle=True, seed=None):
        return VolumeLoaderIterator(
            volume_data_loader, self,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed
        )

    def standardize(self, x):
        """standardize inputs including featurewise/samplewise
        center/normalization, etc."""
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.pixelwise_center:
            if self.pixel_mean is not None:
                x -= self.pixel_mean
        if self.pixelwise_normalization:
            if self.pixel_bounds is not None:
                x = pixelwise_normalize(x, self.pixel_bounds)
        pass

    def random_transform(self, x):
        """random transforms including flipping"""
        pass


class VolumeLoaderIterator(Iterator):

    def __init__(self, volume_data_loader, volume_data_generator,
                 class_mode='binary',
                 batch_size=1, shuffle=False, seed=None):
        self.volume_data_loader = volume_data_loader
        self.volume_data_generator = volume_data_generator
        self.class_mode = class_mode

        self.filenames = volume_data_loader.filenames
        self.nb_sample = len(self.filenames)
        self.image_shape = volume_data_loader.image_shape
        self.classes = volume_data_loader.classes

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
            x = self.volume_data_loader.load(fname)
            # augmentation goes here
            x = self.volume_data_generator.standardize(x)
            batch_x[i] = x
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.

        return batch_x, batch_y


class VolumeDataLoader(object):
    """ Helper class for loading images
    # Data structure Overview
    \directory
        \${image_set}
        ${image_set}_labels.csv
    """
    def __init__(self, directory, image_set, image_format='dcm',
                 split='train', test_size=0.2, random_state=42,
                 target_size=(448, 448, 448), dim_ordering='default',
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
        white_list_formats = {'dcm', 'nii', 'npy'}
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
        """Image series load method to be overwritten.
        # Arguments
            fn: filename of the image (without extension suffix)
        # Returns
            arr: 4D numpy array of shape self.target_size with channels
        """
        pass

    def save(self, arr):
        pass


class DCMDataLoader(VolumeDataLoader):
    """Load dcm volume data using Kaggle kernel by Guido Zuidhof
    See https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    for details
    """

    def load(self, fn):
        img_path = os.path.join(self.directory, fn)
        patient = load_scan(img_path)
        patient_pixels = get_pixels_hu(patient)
        pix_resampled, spacing = resample(patient_pixels, patient, [1, 1, 1])
        # pad to target_size
        resampled_img = pad_to_shape(pix_resampled, self.target_size)
        # todo
        arr = img_to_array(resampled_img, self.dim_ordering)

        return arr


def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2]
                                 - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation
                                 - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(
                np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness]
                       + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor,
                                             mode='nearest')

    return image, new_spacing


class NIIDataLoader(VolumeDataLoader):

    def load(self, fn):
        img_path = os.path.join(self.image_dir,
                                'Axial_{}.nii.gz'.format(fn))
        if not os.path.exists(img_path):
            raise IOError('Image {} does not exist.'.format(img_path))
        img = ni.load_image(img_path)
        arr = img_to_array(img, self.dim_ordering)

        return arr


class NPYDataLoader(VolumeDataLoader):

    def load(self, fn):
        img_path = os.path.join(self.image_dir,
                                '{}.npy'.format(fn))
        img = np.load(img_path)
        pad_img = pad_to_shape(img, self.target_size)
        arr = img_to_array(pad_img, self.dim_ordering)

        return arr
