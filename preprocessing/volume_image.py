from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os
import abc
import scipy
import numpy as np
import pandas as pd
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import Iterator
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


class VolumeImageDataGenerator(object):

    def __init__(self,
                 image_shape=None,
                 target_shape=None,
                 samplewise_resample=None,
                 pixelwise_center=None,
                 pixel_mean=None,
                 pixelwise_normalization=None,
                 pixel_bounds=None,
                 samplewise_center=None,
                 samplewise_std_normalization=None,
                 imlearn_resampler=None,
                 preprocessing_function=None,
                 data_augmentation=None,
                 dim_ordering='default'):
        """
        # Arguments
            image_shape: 3D/4D image shape tuple in the format of
                (dim1, dim2, dim3) for single-channel images,
                (dim1, dim2, dim3, channel) for multi-channel images
                with tensorflow backend
                or (channel, dim1, dim2, dim3) with theano backend.
                It will be overwritten by target_shape when input images
                are of different sizes.
            target_shape: 3D/4D target size for the variant shape images.
                (dim1, dim2, dim3) for single-channel images,
                (dim1, dim2, dim3, channel) for multi-channel images
                with tensorflow backend
                or (channel, dim1, dim2, dim3) with theano backend.
                It will overwrite image_shape if target_shape is set.
            samplewise_resample: True if samplewise resampling
                (subsampling and/or interpolation) is required
                to achieve the target_shape, and False otherwise
            samplewise_center:
            samplewise_std_normalization:
            pixelwise_center: substract the pixel mean of the whole dataset
            pixel_mean: dataset_mean (e.g. 0.25 for LUNA2016)
            pixelwise_normalization: scaled by pixel_bounds
            pixel_bounds: tuple of pixel bounds (min_bound, max_bound)
            imlearn_resampler: resampler for imbalanced dataset {None, 'rus'}
            data_augmentation: {True, False, None}
            dim_ordering: Keras image_dim_ordering {'tf', 'th', 'default'}
        """
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

        # image_shape
        if image_shape and len(image_shape) not in {3, 4}:
            raise ValueError('image_shape needs to be 3 dimensions for '
                             'single-channel volume image data, or 4 '
                             'dimensions for multi-channel data. '
                             'Got {} dimensions'.format(
                                 len(image_shape)))
        if image_shape and len(image_shape) == 3:
            if self.dim_ordering == 'tf':
                self.image_shape = tuple(image_shape) + (1,)
            else:
                self.image_shape = (1,) + tuple(image_shape)
        elif image_shape and len(image_shape) == 4:
            self.image_shape = self.image_shape
        # target_shape
        if target_shape and len(target_shape) not in {3, 4}:
            raise ValueError('target_shape needs to be 3 dimensions for '
                             'single-channel volume image data, or 4 '
                             'dimensions for multi-channel data. '
                             'Got {} dimensions'.format(
                                 len(target_shape)))
        if target_shape and len(target_shape) == 3:
            if self.dim_ordering == 'tf':
                self.target_shape = tuple(target_shape) + (1,)
            else:
                self.target_shape = (1,) + tuple(target_shape)
            self.image_shape = self.target_shape
        elif target_shape and len(target_shape) == 4:
            self.target_shape = self.target_shape
            self.image_shape = self.target_shape
        self.samplewise_resample = samplewise_resample

        self.pixelwise_center = pixelwise_center
        self.pixel_mean = pixel_mean
        self.pixelwise_normalization = pixelwise_normalization
        self.pixel_bounds = pixel_bounds
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
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.pixelwise_normalization:
            if self.pixel_bounds is not None:
                x = pixelwise_normalize(x, self.pixel_bounds)
        if self.pixelwise_center:
            if self.pixel_mean is not None:
                x -= self.pixel_mean
        if self.target_shape and not self.samplewise_resample:
            x = to_shape(x, self.target_shape,
                    constant_values=-1024)
        elif self.target_shape and self.samplewise_resample:
            x = samplewise_resample(x, self.target_shape)
        if self.samplewise_center:
            x -= np.mean(x, axis=self.channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=self.channel_axis, keepdims=True) + 1e-7)
        return x

    def random_transform(self, x):
        """random transforms including flipping"""
        pass


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

        super(VolumeImageLoaderIterator, self).__init__(self.nb_sample, batch_size,
                                                   shuffle, seed)
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
            batch_y = to_categorical(self.classes[index_array], self.nb_classes)
        elif self.class_mode == 'segmentation':
            batch_x = np.zeros((current_batch_size,) + self.image_shape,
                           dtype=np.int16)
            for i, j in enumerate(index_array):
                patient = self.classes[j]
                y = self.volume_image_data_loader.load_labels(patient)
                y = self.volume_image_data_generator.standardize(y)
                batch_y[i] = y

        return batch_x, batch_y


class VolumeImageDataLoader(object):
    """ Helper class for loading images and labels
    # Arguments
        image_dir: directory of volume images
        label_dir: directory
    """
    def __init__(self, image_dir, label_dir, image_format='npy',
                 split='train', test_size=0.2, random_state=42,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='dcm'):

        if not os.path.exists(image_dir):
            raise IOError('Directory {} does not exist. Please provide a '
                          'valid directory.'.format(image_dir))
        self.image_dir = image_dir

        if not os.path.exists(label_dir):
            raise IOError('Directory {} does not exist. Please provide a '
                          'valid directory.'.format(label_dir))
        self.label_dir = label_dir

        if split not in {'train', 'val', 'trainval', 'test', 'predict'}:
            raise ValueError('dataset split must be in '
                             '{"train", "val", "trainval", "test", "predict"} '
                             'Got {}'.format(split))
        self.split = split
        self.test_size = test_size

        self.patients = self.load_patients()
        self.classes = self.load_classes()
        if not len(self.patients) == len(self.classes):
            raise ValueError('Number of patients needs to match number of '
                    'classes. Got {} vs. {}'.format(len(self.patients),
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
        patients_train, patients_test, classes_train, classes_test \
            = train_test_split(self.patients, self.classes,
                               test_size=test_size, random_state=random_state)
        if self.split == 'train':
            self.patients = patients_train
            self.classes = classes_train
        elif self.split == 'val':
            self.patients = patients_test
            self.classes = classes_test
        else:
            pass

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

    @abc.abstractmethod
    def load_patients(self):
        """Image patients loading method. (Abstract)
        (the following is an example)
        """
        try:
            df_labels = pd.read_csv(self.label_dir)
        except IOError:
            raise
        return df_labels['id'].values

    @abc.abstractmethod
    def load_classes(self):
        """Image classes, labels or label pointers loading method (Abstract)
        where
            classes: image class (binary, categorical or sparse)
            labels: sparse regression labels, e.g. bounding boxes
            label pointers: pointers of dense labels, e.g. segmentation
        # Returns
            classes: vector of classes/labels/pointers
        """
        try:
            df_labels = pd.read_csv(self.label_dir)
        except IOError:
            raise
        return df_labels['cancer'].values.astype('int')

    @abc.abstractmethod
    def load(self, p):
        """Image series loading method given a filename (Abstract)
        # Arguments
            p: patient id
        # Returns
            arr: 4D numpy array of shape self.image_shape
        """
        img_path = os.path.join(self.image_dir,
                                '{}.{}'.format(p, self.image_format))
        img = np.load(img_path)
        arr = img_to_array(img, self.dim_ordering)

        return arr

    @abc.abstractmethod
    def save(self, arr):
        pass

    @abc.abstractmethod
    def load_labels(self, pr):
        """Dense labels loading methods given pointers to them (Abstract)
        # Arguments
            p: pointers to the dense labels
        # Returns
            arr: 4D array of dense labels
        """
        pass


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
    # Numpy array x has format (dim1, dim2, dim3, channel)
    # or (channel, dim1, dim2, dim3)
    # nipy image has format (I don't know)
    if isinstance(img, np.ndarray):
        x = img.astype(K.floatx())
    else:
        x = img.get_data().astype(K.floatx())
    if len(x.shape) == 4:
        if dim_ordering == 'th':
            x = x.transpose(3, 0, 1, 2)
    elif len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x[np.newaxis, ...]
        else:
            x = x[..., np.newaxis]
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def to_shape(arr, target_shape,  constant_values=0):
    """Pad and/or crop arrays to the target_shape
    """
    # pad
    padded = pad(arr, target_shape)
    # crop
    if padded.shape != target_shape:
        cropped = crop(padded, target_shape)
    else:
        cropped = padded
    return cropped


def samplewise_resample(self, image, target_shape, mode='nearest'):
    """Resample the image to target_shape
    """
    resize_factor = image.shape / np.array(target_shape)
    resampled = scipy.ndimage.interpolation.zoom(image, resize_factor,
                                             mode=mode)
    return resampled


def pad(arr, target_shape, constant_values=0):
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
    padded = np.pad(arr, pad_width=npad,
                     mode='constant',
                     constant_values=constant_values)
    return padded


def crop(arr, target_shape):
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

