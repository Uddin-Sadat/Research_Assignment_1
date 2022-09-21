import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
import argparse
import cv2

def show_slices(slices, x, y):
    fig, axes = plt.subplots(1, len(slices), figsize=(x,y))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        
def overlay_mask(image_slices, mask_slices, x, y):
    fig, axes = plt.subplots(1, len(image_slices), figsize=(x,y))
    for i, (image, mask) in enumerate(zip(image_slices, mask_slices)):
        masked = cv2.bitwise_and(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), mask=cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        axes[i].imshow(masked, cmap="gray", origin="lower")
        

image = nib.load('./ProstateX-0000/ADC.nii.gz')
mask = nib.load('./ProstateX-0000/PM.nii.gz')

image_data = image.get_fdata()
mask_data = mask.get_fdata()
x = image.shape[0]
y = image.shape[1]

show_slices([image_data[:,:,i] for i in range(0,image.shape[2]-1)], image.shape[0], image.shape[1])
show_slices([mask_data[:,:,i] for i in range(0,mask.shape[2]-1)], image.shape[0], image.shape[1])
overlay_mask([image_data[:,:,i] for i in range(0,image.shape[2]-1)], [mask_data[:,:,i] for i in range(0,mask.shape[2]-1)], image.shape[0], image.shape[1])