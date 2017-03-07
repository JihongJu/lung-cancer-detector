import os
import dicom
import scipy
import scipy.ndimage
import numpy as np

from multiprocessing import Pool, cpu_count

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

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
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[2,2,2]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def preprocessing(path):
    patient = load_scan(path)
    patient_pixels = get_pixels_hu(patient)
    pix_resampled, spacing = resample(patient_pixels, patient, [2,2,2])
    print("HU converting: ({}, {}) "
          "--> ({}, {})".format(np.amin(patient[0].pixel_array),
                                np.amax(patient[0].pixel_array),
                                np.amin(patient_pixels),
                                np.amax(patient_pixels)))
    print("Shape resampling: {} --> {}".format(patient_pixels.shape,
                                               pix_resampled.shape))
    return pix_resampled


TRAINVAL = 'stage1'
INPUT_FOLDER = '../data/data-science-bowl/dcm/{}'.format(TRAINVAL)
OUTPUT_FOLDER = '../data/data-science-bowl/npy/{}'.format(TRAINVAL)
patients = os.listdir(INPUT_FOLDER)

def f(p):
    img_path = os.path.join(INPUT_FOLDER, p)
    pix_resampled = preprocessing(img_path)
    output_path = os.path.join(OUTPUT_FOLDER, p)
    np.save(output_path, pix_resampled)
    return pix_resampled.shape

nb_cpus = cpu_count()
pool = Pool(processes=nb_cpus)
results = pool.map(f, patients)
