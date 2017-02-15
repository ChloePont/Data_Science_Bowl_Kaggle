import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy
import SimpleITK
from sklearn.cluster import KMeans
from skimage import morphology, measure
import scipy

# Define function to import scans for one patient
def import_img_series(path):
    reader = SimpleITK.ImageSeriesReader()
    filenamesDICOM = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(filenamesDICOM)
    return reader.Execute()

# Define funcion to load in DICOM images
def load_images(patient_id, path=''):
    path = path + patient_id
    img = import_img_series(path)
    return img


# Define a fuction to remove image noise
def remove_noise(img):
    imgSmooth = SimpleITK.CurvatureFlow(image1=img, timeStep=0.125, numberOfIterations=5)
    return imgSmooth


# Define a function to reshape images
def resample(img):
    # Determine current pixel spacing
    size = np.array([img.GetSize()[2], img.GetSize()[1], img.GetSize()[0]])

    resize_factor = [float(84),float(64),float(64)] / size

    arr = SimpleITK.GetArrayFromImage(img)
    arr_resampled = scipy.ndimage.interpolation.zoom(arr, resize_factor)

    arr_resampled = arr_resampled[10:-10]

    return arr_resampled


# Segment the lungs
def segment_lungs(arr):
    # Produce a satisfactory separation of regions for both types of images
    # and eliminate the black halo in some images
    middle = arr[5:60, 10:45,10:45]
    mean = np.mean(middle)
    max = np.max(arr)
    min = np.min(arr)
    #move the underflow bins
    arr[arr==max]=mean
    arr[arr==min]=mean
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_arr = np.where(arr<threshold,1.0,0.0)  # threshold the image

    # Use erosion and dilation to fill in the incursions into the lungs region by radio-opaque tissue
    # and select the regions based on the bounding box sizes of each region
    eroded = morphology.erosion(thresh_arr)
    dilation = morphology.dilation(eroded)
    labels = measure.label(dilation)
    label_vals = np.unique(labels)

    # Cutting non-ROI regions
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[4]-B[1]<60 and B[5]-B[2]<60 and B[1]>5 and B[2]<60: # Check thresholds as the cuts applied to each region bounding box were determined empirically and seem to work well for the LUNA data, but may not be generally applicable
            good_labels.append(prop.label)
    mask = np.ndarray(thresh_arr.shape, dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask) # one last dilation

    # renormalizing the masked image (in the mask region)
    #
    new_mean = np.mean(arr[mask>0])
    new_std = np.std(arr[mask>0])
    #
    #  Pushing the background color up to the lower end
    #  of the pixel range for the lungs
    #
    old_min = np.min(arr)       # background color
    arr[arr==old_min] = new_mean-1.2*new_std   # resetting backgound color
    arr = arr-new_mean
    arr = arr/new_std

    return arr*mask


# Final function to get segmented lungs out of folder path
# Denoising hasn't been included so far
def get_lungs_arr(patient_id, path=''):
    img = load_images(patient_id, path)
    resampled_arr = resample(img)
    segmented_lungs = segment_lungs(resampled_arr)
    return segmented_lungs

# Define a function to display several slices of the 4 sequences 

def sitk_show_slices(img, margin=0.05, dpi=40, axis='off', size=(10,10)):
    length = np.sqrt(img.GetSize()[2])
    length = int(np.ceil(length))
    fig, im = plt.subplots(length, length, figsize=size)
        
    for i in range(0, img.GetSize()[2]-length, length):
        imgs = [img[:,:,j] for j in range(i, i+length)]
        for j in range(length):
            nda = SimpleITK.GetArrayFromImage(imgs[j])
            spacing = imgs[j].GetSpacing()
            figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
            extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)

            plt.set_cmap("gray")
            im[i/length,j].imshow(nda,extent=extent,interpolation=None)
            im[i/length,j].axis(axis)
        
    last_imgs = [img[:,:,j] for j in range(length**2-length, img.GetSize()[2])] 
    for j in range(length - (length**2 - img.GetSize()[2])):
        nda = SimpleITK.GetArrayFromImage(last_imgs[j])
        spacing = last_imgs[j].GetSpacing()
        figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
        extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)

        plt.set_cmap("gray")
        im[length-1, j].imshow(nda,extent=extent,interpolation=None)
        im[length-1, j].axis(axis)

    fig.show()
