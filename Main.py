import cv2
import numpy as np
from PIL import Image
from Mask import apply_mask

cap_left = cv2.VideoCapture('Resources/synced_left.mp4')
cap_right = cv2.VideoCapture('Resources/synced_right.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (3840, 1080))

H = np.load('H.npy')
mask = Image.open("mask.png")

while(cap_left.isOpened() and cap_right.isOpened()):
    ret, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    if ret and ret2:
        right = cv2.warpPerspective(frame_right,H,(frame_left.shape[1] + frame_right.shape[1], frame_left.shape[0]))
        left = np.zeros(right.shape)
        left[0:frame_left.shape[0], 0:frame_left.shape[1]] = frame_left
        dst = apply_mask(right, left, mask)
        out.write(dst)
    else:
        break

cap_left.release()
cap_right.release()
out.release()