import os
import dicom
import numpy as np
import pandas as pd
import nipy as ni
import nibabel as nib
from nipy.core.api import Image, vox2mni


def load_data(trainval='stage1', img_format='dicom',
              target_size=(512, 512, 128)):
    """Loads datasets of Data Science Bowl 2017
    # Arguments
        trainval: specify which set of images for train and validation
                  (stage1, sample_images)
    # Returns
        Tuple of Numpy arrays: `(x, y)`.
    Overview of the data directory structure:
    \root
        \data
            \data-science-bowl
                \dicom
                    \sample_images
                    \stage1
                \nii
                    \sample_images
                    \stage1
                sample_images_labels.csv
                stage1_labels.csv
    """
    dirname = 'data-science-bowl'
    path = os.path.join('data', dirname)

    # load images
    df_labels = pd.read_csv(path+"/{}_labels.csv".format(trainval))

    nb_series = len(df_labels)
    x = np.zeros((nb_series,)+target_size, dtype='float16')

    for s_id, series in zip(df_labels.index, df_labels['id']):
        # load image
        if img_format == 'nii':
            img_path = os.path.join(path, img_format, trainval,
                                    'Axial_{}.nii.gz'.format(series))
            img = ni.load_image(img_path)
        elif img_format == 'dicom':
            img_path = os.path.join(path, img_format, trainval, series)
            img_filenames = os.listdir(img_path)
            nb_slices = len(img_filenames)
            x_series = np.zeros((512, 512, nb_slices), dtype='int16')
            for f_id, filename in enumerate(img_filenames):
                dc = dicom.read_file(os.path.join(img_path, filename))
                x_series[..., f_id] = dc.pixel_array
            img = Image(x_series, vox2mni(np.eye(4)))
        # Coeregistration / normalization
        # See: http://nipy.org/nipy/devel/code_discussions/usecases/images.html

        print img.shape
        print type(img)

    x = []
    y = []
    return x, y
