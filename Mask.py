import cv2
import numpy as np
from ColorCorrection import hist_match
from ColorCorrection import color_transfer
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
from PIL import Image

img_right = cv2.imread('Resources/right.jpg')
img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

img_left = cv2.imread('Resources/left.jpg')
img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

H = np.load('H.npy')

right = cv2.warpPerspective(img_right, H, (img_left.shape[1] + img_right.shape[1], img_left.shape[0]))
mask = cv2.warpPerspective(np.ones(img_right.shape), H, (img_left.shape[1] + img_right.shape[1], img_left.shape[0]))   

mask = (mask != 0)*1.0
left = np.zeros(right.shape)
left[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
ib = left != 0
i = mask * ib
#i2 = (1 - i).astype(bool)

end = img_left.shape[1]

def interpolate(start, end, ind):
    return (ind - start) / (end - start) 

for ind, flag in enumerate(range(mask.shape[1])):
    if np.any(i[:,ind, 0]) == True:
        start = ind
        break

for ind, flag in enumerate(range(mask.shape[1])):
    if ind <= end and ind >= start:
        mask[:, ind, :] = mask[:, ind, :] * interpolate(start, end, ind)

mask = mask[:,:,0]
Image.fromarray(mask * 255).convert("L").save("mask.png")

def apply_mask(right, left, mask):
    right = Image.fromarray(right.astype(np.uint8))
    left = Image.fromarray(left.astype(np.uint8))
    return np.array(Image.composite(right, left, mask))