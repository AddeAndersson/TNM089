import cv2
import numpy as np
from ColorCorrection import hist_match
from ColorCorrection import color_transfer
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
from PIL import Image
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

#img_right = cv2.imread('Resources/right.jpg')
#img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

#img_left = cv2.imread('Resources/left.jpg')
#img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

#img_right = img_right.astype(np.uint8)
#img_left = img_left.astype(np.uint8)

img_left = np.array(Image.open('Resources/left.jpg'))
img_right = np.array(Image.open('Resources/right.jpg'))

# Color correction
img_left_cc = hist_match(img_left, img_right)
img_left_cc = color_transfer(img_right, img_left)

img_left_cc = match_histograms(img_left, img_right, multichannel=True)

H = np.load('H.npy')

out = cv2.warpPerspective(img_right, H, (img_left.shape[1] + img_right.shape[1], img_left.shape[0]))
out[0:img_left.shape[0], 0:img_left.shape[1]] = img_left_cc

plt.imshow(out.astype(np.int), cmap='gray')
plt.show()

# Histogram plot
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

for i, img in enumerate((img_left, img_right, img_left_cc)):
    for c, c_color in enumerate(('red', 'green', 'blue')):
        img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
        axes[c, i].plot(bins, img_hist / img_hist.max())
        img_cdf, bins = exposure.cumulative_distribution(img[..., c])
        axes[c, i].plot(bins, img_cdf)
        axes[c, 0].set_ylabel(c_color)

axes[0, 0].set_title('Source')
axes[0, 1].set_title('Reference')
axes[0, 2].set_title('Matched')

plt.tight_layout()
plt.show()