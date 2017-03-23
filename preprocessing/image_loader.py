import os
import scipy
import dicom
import numpy as np
import pandas as pd
import nipy as ni
import keras.backend as K
from preprocessing.volume_image import (
        VolumeImageDataLoader)


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


class DSBDataLoader(VolumeImageDataLoader):

    def get_patients(self):
        df_labels = pd.read_csv(self.label_dir)
        return df_labels['id'].values

    def get_classes(self):
        df_labels = pd.read_csv(self.label_dir)
        return df_labels['cancer'].values.astype('int')


class NPYDataLoader(DSBDataLoader):

    def load(self, p):
        img_path = os.path.join(self.image_dir,
                                '{}.npy'.format(p))
        img = np.load(img_path)
        arr = img_to_array(img, self.dim_ordering)

        return arr


class NIIDataLoader(DSBDataLoader):

    def load(self, p):
        img_path = os.path.join(self.image_dir,
                                'Axial_{}.nii.gz'.format(p))
        if not os.path.exists(img_path):
            raise IOError('Image {} does not exist.'.format(img_path))
        img = ni.load_image(img_path)
        arr = img_to_array(img, self.dim_ordering)

        return arr


class DCMDataLoader(DSBDataLoader):
    """Load dcm volume data using Kaggle kernel by Guido Zuidhof."""

    def load(self, p):
        img_path = os.path.join(self.directory, p)
        patient = self.load_scan(img_path)
        patient_pixels = self.get_pixels_hu(patient)
        pix_resampled, spacing = self.resample(patient_pixels, patient,
                                               [1, 1, 1])

        arr = img_to_array(pix_resampled, self.dim_ordering)

        return arr

    def load_scan(self, path):
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

    def get_pixels_hu(self, slices):
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

    def resample(self, image, scan, new_spacing=[1, 1, 1]):
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
