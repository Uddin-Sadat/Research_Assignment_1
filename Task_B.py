import matplotlib.pyplot as plt
from PIL import Image as im
import nibabel as nib
import numpy as np
import argparse
import cv2

def outline_images(image_slices, mask_slices, x, y):
    fig, axes = plt.subplots(1, len(image_slices), figsize=(x,y))
    for i, (image, mask) in enumerate(zip(image_slices, mask_slices)):
        mask = mask.astype(np.uint8)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        image = cv2.UMat(image)
        for c in cnts:
            cv2.drawContours(image, [c], -1, (0, 0, 0), thickness=2)
        img_array = cv2.UMat.get(image)
        axes[i].imshow(img_array, cmap="gray", origin="lower")
        filename = "Image" + str(i) + ".png"
        plt.imsave(filename, img_array, cmap="gray")
    

image = nib.load('./ProstateX-0000/ADC.nii.gz')
mask = nib.load('./ProstateX-0000/PM.nii.gz')

image_data = image.get_fdata()
mask_data = mask.get_fdata()
x = image.shape[0]
y = image.shape[1]

outline_images([image_data[:,:,i] for i in range(0,image.shape[2]-1)], [mask_data[:,:,i] for i in range(0,mask.shape[2]-1)], image.shape[0], image.shape[1])