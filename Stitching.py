import cv2
import numpy as np
from KNN import KNN
import matplotlib.pyplot as plt
from random import randrange
from ShowMatches import show_corresp 
from Homography import Homography
import time

img_right = cv2.imread('Resources/right.jpg')
img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
img_r = cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY)

img_left = cv2.imread('Resources/left.jpg')
img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_l = cv2.cvtColor(img_left,cv2.COLOR_RGB2GRAY)

# Create SIFT and extract features
sift = cv2.xfeatures2d.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img_r, None)
kp2, des2 = sift.detectAndCompute(img_l, None)

# Find KNN matches and validate with ratio test
#knn_solver = KNN(des1, des2, 2)
#matches = knn_solver.solve()

# Find KNN matches (correct version)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m in matches:
     if m[0].distance < 0.5*m[1].distance:         
     	good.append(m)
matches = np.asarray(good)


# Extract keypoints' coordinates
#valid_kp1 = []
#valid_kp2 = []
#for match in matches:
#    valid_kp1.append(np.array(kp1[match.index_1].pt))
#    valid_kp2.append(np.array(kp2[match.indices_2[0]].pt))
#valid_kp1 = np.array(valid_kp1).T
#valid_kp2 = np.array(valid_kp2).T

# Find homography
#H = Homography(valid_kp1, valid_kp2, 10000)
#with open('H.npy', 'wb') as f:
    #np.save(f, H)
#print(H)

# Visualize matches
#show_corresp(img_l, img_r, valid_kp2[0:100], valid_kp1[0:100], vertical=0)
#plt.show()

# Find homography (correct version)

src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
t0 = time.clock()
H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0, maxIters=10000)
t1 = time.clock() - t0
print("Time elapsed - OpenCV find homography: ", t1)
print("Homography: \n", H)
'''
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