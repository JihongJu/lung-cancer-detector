"""Volume Image generator."""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os
import abc
import scipy
import numpy as np
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import Iterator
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


class VolumeImageDataGenerator(object):
    """Volume image data generator
    # Arguments
        image_shape: 3D/4D image shape tuple in the format of
            (dim1, dim2, dim3) for single-channel images,
            (dim1, dim2, dim3, channel) for multi-channel images
            with tensorflow backend
            or (channel, dim1, dim2, dim3) with theano backend.
        image_resample: True if samplewise resampling
            (subsampling and/or interpolation) is required
            to achieve the target_shape, and False otherwise
        samplewise_center:
        samplewise_std_normalization:
        voxelwise_center: substract the voxel mean of the whole dataset
        voxel_mean: dataset_mean (e.g. 0.25 for LUNA2016)
        voxelwise_normalization: scaled by voxel_bounds
        voxel_bounds: tuple of voxel bounds (min_bound, max_bound)
        imlearn_resampler: resampler for imbalanced dataset {None, 'rus'}
        data_augmentation: {True, False, None}
        dim_ordering: Keras image_dim_ordering {'tf', 'th', 'default'}
    """

    def __init__(self,
                 image_shape,
                 image_resample=None,
                 voxelwise_center=None,
                 voxel_mean=None,
                 voxelwise_std_normalization=None,
                 voxelwise_std=None,
                 voxelwise_normalization=None,
                 voxel_bounds=None,
                 samplewise_center=None,
                 samplewise_std_normalization=None,
                 imlearn_resampler=None,
                 preprocessing_function=None,
                 data_augmentation=None,
                 dim_ordering='default'):

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

        # fixed image_shape
        if len(image_shape) not in {3, 4}:
            raise ValueError('image_shape needs to be 3 dimensions for '
                             'single-channel volume image data, or 4 '
                             'dimensions for multi-channel data. '
                             'Got {} dimensions'.format(
                                 len(image_shape)))
        if len(image_shape) == 3:
            if self.dim_ordering == 'tf':
                self.image_shape = tuple(image_shape) + (1,)
            else:
                self.image_shape = (1,) + tuple(image_shape)
        elif len(image_shape) == 4:
            self.image_shape = self.image_shape

        self.image_resample = image_resample
        self.voxelwise_center = voxelwise_center
        self.voxel_mean = voxel_mean
        self.voxelwise_std_normalization = voxelwise_std_normalization
        self.voxelwise_std = voxelwise_std
        self.voxelwise_normalization = voxelwise_normalization
        self.voxel_bounds = voxel_bounds
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization

        if imlearn_resampler and imlearn_resampler not in {'rus'}:
            raise ValueError("imlearn_resampler must be in {'rus'}.")
        self.imlearn_resampler = imlearn_resampler
        self.preprocessing_function = preprocessing_function
        self.data_augmentation = data_augmentation

    def flow_from_loader(self, volume_image_data_loader,
                         class_mode='binary', nb_classes=None,
                         batch_size=32, shuffle=True, seed=None):
        return VolumeImageLoaderIterator(
            volume_image_data_loader, self,
            class_mode=class_mode,
            nb_classes=nb_classes,
            imlearn_resampler=self.imlearn_resampler,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed
        )

    def standardize(self, x):
        """standardize inputs including featurewise/samplewise
        center/normalization, etc."""
        if not self.image_resample:
            x = to_shape(x, self.image_shape, constant_values=-1024)
        elif self.image_resample:
            x = resample(x, self.image_shape)

        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.voxelwise_normalization:
            if self.voxel_bounds is not None:
                x = voxelwise_normalize(x, self.voxel_bounds)
        if self.voxelwise_center:
            if self.voxel_mean is not None:
                x -= self.voxel_mean
        if self.voxelwise_std_normalization:
            x /= (self.voxelwise_std + 1e-7)
        if self.samplewise_center:
            x -= np.mean(x, axis=self.channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=self.channel_axis, keepdims=True) + 1e-7)
        return x

    def random_transform(self, x):
        """Random transforms including flipping."""
        pass

    def standardize_dense_labels(self, y):
        """Standardize dense labels."""
        if self.image_shape and not self.image_resample:
            y = to_shape(y, self.image_shape, constant_values=0)
        elif self.image_shape and self.image_resample:
            y = resample(y, self.image_shape, mode='nearest')
        return y


class VolumeImageLoaderIterator(Iterator):

    def __init__(self, volume_image_data_loader, volume_image_data_generator,
                 class_mode='binary', nb_classes=None, imlearn_resampler=None,
                 batch_size=1, shuffle=False, seed=None):
        self.volume_image_data_loader = volume_image_data_loader
        self.volume_image_data_generator = volume_image_data_generator
        self.class_mode = class_mode
        self.nb_classes = nb_classes

        self.patients = volume_image_data_loader.patients
        self.classes = volume_image_data_loader.classes
        self.nb_sample = len(self.patients)
        self.image_shape = volume_image_data_generator.image_shape

        # Random under sampler for imbalanced class
        self.imlearn_resampler = imlearn_resampler
        if imlearn_resampler:
            self.patients_imbalanced = self.patients.copy()
            self.classes_imbalanced = self.classes.copy()

        if imlearn_resampler == 'rus':
            self.resampler = RandomUnderSampler()

        super(VolumeImageLoaderIterator, self).__init__(self.nb_sample,
                                                        batch_size,
                                                        shuffle,
                                                        seed)

    def reset(self):
        # ensure self.batch_index is 0
        self.batch_index = 0
        # resampling imbalanced dataset
        if self.imlearn_resampler:
            self.patients, self.classes = self.resampler.fit_sample(
                 self.patients_imbalanced, self.classes_imbalanced)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        batch_x = np.zeros((current_batch_size,) + self.image_shape,
                           dtype=K.floatx())
        for i, j in enumerate(index_array):
            patient = self.patients[j]
            x = self.volume_image_data_loader.load(patient)
            # standardize goes here
            x = self.volume_image_data_generator.standardize(x)
            batch_x[i] = x

        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = to_categorical(self.classes[index_array],
                                     self.nb_classes)
        elif self.class_mode == 'segmentation':
            label_shape = list(self.image_shape)
            label_shape[self.volume_image_data_generator.channel_axis - 1] \
                = self.nb_classes
            label_shape = tuple(label_shape)
            batch_y = np.zeros((current_batch_size,) + label_shape,
                               dtype=np.int8)
            for i, j in enumerate(index_array):
                patient = self.classes[j]
                y = self.volume_image_data_loader.load_label(patient)
                y = self.volume_image_data_generator\
                        .standardize_dense_labels(y)
                y = to_categorical(y, self.nb_classes).reshape(label_shape)
                batch_y[i] = y

        return batch_x, batch_y


class VolumeImageDataLoader(object):
    """ Helper class for loading images and labels
    # Arguments
        image_dir: directory of volume images
        label_dir: directory
    """
    def __init__(self, image_dir, label_dir=None, image_format='npy',
                 split=None, test_size=None, random_state=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='dcm'):

        if not os.path.exists(image_dir):
            raise IOError('Directory {} does not exist. Please provide a '
                          'valid directory.'.format(image_dir))
        self.image_dir = image_dir

        if label_dir and not os.path.exists(label_dir):
            raise IOError('Directory {} does not exist. Please provide a '
                          'valid directory.'.format(label_dir))
        self.label_dir = label_dir

        if split not in {'train', 'val', 'trainval', 'test', 'predict'}:
            raise ValueError('dataset split must be in '
                             '{"train", "val", "trainval", "test", "predict"} '
                             'Got {}'.format(split))
        self.split = split
        self.test_size = test_size

        self.patients, self.classes = self.get_patients_classes()
        if not len(self.patients) == len(self.classes) \
                and not self.split == 'test':
            raise ValueError('Number of patients needs to match number of '
                             'classes. Got {} vs. {}'.format(
                                 len(self.patients),
                                 len(self.classes)))

        self.image_format = image_format
        white_list_formats = {'dcm', 'nii', 'npy', 'h5', 'mha'}
        if image_format not in white_list_formats:
            raise ValueError('Invalid image format:', image_format,
                             '; expected "dcm", "nii", "mha",'
                             '"npy", "h5".')

        if dim_ordering not in {'default', 'tf', 'th'}:
            raise ValueError('Invalid dim ordering:', dim_ordering,
                             '; expected "default", "tf" or "th".')
        self.dim_ordering = dim_ordering
        if dim_ordering == 'default':
            self.dim_ordering = K.image_dim_ordering()

        # split train test or keep all
        if test_size:
            patients_train, patients_test, classes_train, classes_test \
                = train_test_split(self.patients, self.classes,
                                   test_size=test_size,
                                   random_state=random_state)
            if self.split == 'train':
                self.patients = patients_train
                self.classes = classes_train
            elif self.split == 'val':
                self.patients = patients_test
                self.classes = classes_test

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

    @abc.abstractmethod
    def get_patients_classes(self):
        """Get patients and classes altogethre (Abstract)."""
        patients = self.get_patients()
        classes = self.get_classes()
        return patients, classes

    @abc.abstractmethod
    def get_patients(self):
        """Image patients loading method. (Abstract)
        (the following is an example)
        """
        return

    @abc.abstractmethod
    def get_classes(self):
        """Image classes, labels or label pointers loading method (Abstract)
        where
            classes: image class (binary, categorical or sparse)
            labels: sparse regression labels, e.g. bounding boxes
            label pointers: pointers of dense labels, e.g. segmentation
        # Returns
            classes: vector of classes/labels/pointers
        """
        return

    @abc.abstractmethod
    def load(self, p):
        """Image series loading method given a filename (Abstract).
        # Arguments
            p: patient id
        # Returns
            arr: 4D numpy array of shape self.image_shape
        """
        return

    @abc.abstractmethod
    def save(self, arr):
        return

    @abc.abstractmethod
    def load_label(self, pr):
        """Dense labels loading methods given pointers to them (Abstract).
        # Arguments
            p: pointers to the dense labels
        # Returns
            arr: 4D array of dense labels
        """
        return


def voxelwise_normalize(image, voxel_bounds):
    if len(voxel_bounds) != 2:
        raise ValueError("Expected tuple of voxel bounds "
                         "(min_bound, max_bound). Got {}".format(voxel_bounds))
    min_bound = voxel_bounds[0]
    max_bound = voxel_bounds[1]
    image = (image - min_bound) / (max_bound - min_bound)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def to_shape(arr, target_shape,  constant_values=0):
    """Pad and/or crop arrays to the target_shape."""
    # pad
    padded = pad(arr, target_shape)
    # crop
    if padded.shape != target_shape:
        cropped = crop(padded, target_shape)
    else:
        cropped = padded
    return cropped


def resample(self, image, target_shape, mode='nearest'):
    """Resample the image to target_shape
    """
    resize_factor = image.shape / np.array(target_shape)
    resampled = scipy.ndimage.interpolation.zoom(image, resize_factor,
                                                 mode=mode)
    return resampled


def pad(arr, target_shape, constant_values=0):
    """Pad image to target shape."""
    arr_shape = arr.shape
    npad = ()
    for dim in range(len(arr_shape)):
        diff = target_shape[dim] - arr_shape[dim]
        if diff > 0:
            before = int(diff / 2)
            after = diff - before
        else:
            before = 0
            after = 0
        npad += ((before, after),)
    padded = np.pad(arr, pad_width=npad, mode='constant',
                    constant_values=constant_values)
    return padded


def crop(arr, target_shape):
    """Crop image to targe shape."""
    arr_shape = arr.shape
    ncrop = ()
    for dim in range(len(arr_shape)):
        diff = arr_shape[dim] - target_shape[dim]
        if diff > 0:
            start = int(diff / 2)
            stop = start + target_shape[dim]
            ncrop += np.index_exp[start:stop]
        else:
            ncrop += np.index_exp[:]
    cropped = arr[ncrop]
    return cropped
