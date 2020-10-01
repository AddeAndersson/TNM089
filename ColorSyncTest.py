import cv2
import numpy as np
from ColorCorrection import hist_match
from ColorCorrection import color_transfer
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms

img_right = cv2.imread('Resources/right.jpg')
img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
#img_r = cv2.cvtColor(img_right, cv2.COLOR_RGB2HSV)

img_left = cv2.imread('Resources/left.jpg')
img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
#img_l = cv2.cvtColor(img_left,cv2.COLOR_RGB2HSV)

#img_r[:, :, 2] = hist_match(img_r[:,:,2], img_l[:,:,2])
#img_r = cv2.cvtColor(img_r, cv2.COLOR_HSV2RGB)
#img_l = cv2.cvtColor(img_l, cv2.COLOR_HSV2RGB)

#img_l = match_histograms(img_left, img_right, multichannel=True)
#img_l = color_transfer(img_right, img_left)
img_l = img_left
img_r = img_right

H = np.load('H.npy')

dst = cv2.warpPerspective(img_r, H, (img_left.shape[1] + img_right.shape[1], img_left.shape[0]))     

ia = dst != 0
tmp = np.zeros(dst.shape)
tmp[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
ib = tmp != 0
i = ia * ib
i2 = (1 - i).astype(bool)

#plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
#plt.show()
#plt.figure()
dst[0:img_left.shape[0], 0:img_left.shape[1]] = img_l
dst[i2] = 0
cv2.imwrite('resultant_color_test.jpg',dst)
plt.imshow(dst)
plt.show()