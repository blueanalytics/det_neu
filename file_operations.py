
# file_operations.py
import os
import cv2
from PIL import Image as Img
from PIL import ImageTk
from PIL import Image
import pydicom
import numpy as np

def read_dicom_file(path):
    img = pydicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Img.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show

def read_jpg_file(path):
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Img.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show
