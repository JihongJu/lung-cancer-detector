#!/usr/bin/env python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import nipy as ni
import scipy.ndimage
import matplotlib.pyplot as plt

from scipy import stats
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants 
TRAINVAL = 'stage1'
INPUT_FOLDER = '../data/data-science-bowl/nii/{}'.format(TRAINVAL)
OUTPUT_FOLDER = '../data/data-science-bowl/npy/{}'.format(TRAINVAL)
LABEL_FILE = '../data/data-science-bowl/nii/{}_labels.csv'.format(TRAINVAL)
SPACING_FILE = '../data/data-science-bowl/nii/{}_spacing.csv'.format(TRAINVAL)


def resampling(image, spacing, new_spacing=[2,2,2]):
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image,
                                             real_resize_factor,
                                             mode='nearest')
    print("Spacing change: {} --> {}".format(spacing, new_spacing)) 
    return image, new_spacing



def preprocessing(img_path, img_spacing):
    # load
    img = ni.load_image(img_path)
    # resampling
    patient_pixels = img.get_data().astype('int16')
    patient_pixels[patient_pixels==-3024]=-1024
    patient_pixels[patient_pixels==-2048]=-1024
    pix_resampled = resampling(patient_pixels, img_spacing)
    print("Data range: ({}, {}) --> ({}, {})".format(
        np.amin(patient_pixels),
        np.amax(patient_pixels),
        np.amin(pix_resampled),
        np.amax(pix_resampled)))
    print("Shape resampling: {} --> {}".format(patient_pixels.shape,
                                               pix_resampled.shape))
    return (pix_resampled.shape)



df_spacing = pd.read_csv(LABEL_FILE)
patients = df_spacing['id'].values
spacing = df_spacing['spacing'].values


def f(p, s):
    img_path = os.path.join(INPUT_FOLDER,
                        'Axial_{}.nii.gz'.format(p))
    output_path = os.path.join(OUTPUT_FOLDER, p)
    pix_resampled = preprocessing(img_path, s)
    np.save(output_path, pix_resampled)


pool = Pool(cpu_count())
results = pool.map(f, zip(patients, spacing))
pool.close()
pool.join()