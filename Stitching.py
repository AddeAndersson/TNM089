import cv2
import numpy as np
from KNN import KNN
import matplotlib.pyplot as plt
from random import randrange

img_right = cv2.imread('Resources/right.jpg')
img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
img_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

img_left = cv2.imread('Resources/left.jpg')
img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_l = cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)

# Create SIFT and extract features
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img_r, None)
kp2, des2 = sift.detectAndCompute(img_l, None)

print(des1.shape)
tmp_des1 = des1[0:10000, :]
tmp_des2 = des2[0:10000, :]
# BFMatcher with default params
#bf = cv2.BFMatcher()
knn_solver = KNN(tmp_des1, tmp_des2, 2)
matches = knn_solver.solve()
print(len(matches))
#matches = bf.knnMatch(des1, des2, k=2)

'''

# Apply ratio test
good = []
for m in matches:
     if m[0].distance < 0.5*m[1].distance:         
     	good.append(m)
matches = np.asarray(good)
 	 

# Find homography
if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print(H)
else:
    raise AssertionError("Can't find enough keypoints.")

print("Homography found")


# Transform and output
dst = cv2.warpPerspective(img_right,H,(img_left.shape[1] + img_right.shape[1], img_left.shape[0]))     	
plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
plt.show()
plt.figure()
dst[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
cv2.imwrite('resultant_stitched_panorama.jpg',dst)
plt.imshow(dst)
plt.show()
cv2.imwrite('resultant_stitched_panorama.jpg',dst)
'''