import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import SimpleITK


def sitk_show(img, title=None, margin=0.0, dpi=40, axis='off'):
    nda = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    ax.axis(axis)
    
    if title:
        plt.title(title)
    
    plt.show()


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


# Define function to load in images as intensity arrays
def load_array(patient_id):
    return SimpleITK.GetArrayFromImage(load_images(patient_id))


# Define a fuction to remove image noise
def remove_noise(img):
    imgSmooth = SimpleITK.CurvatureFlow(image1=img, timeStep=0.125, numberOfIterations=5)
    return imgSmooth


def reshape_image(img):
    # Define resample filter to match T1 image dimension and settings
    resample = SimpleITK.ResampleImageFilter()
    resample.SetInterpolator(SimpleITK.sitkBSpline)
    resample.SetSize((512,512,94))

    # Resize all other three images
    img_Resized = resample.Execute(img)

    return img_Resized
