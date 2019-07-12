import numpy as np
import cv2


img = np.zeros((3,512,512), np.uint8)

b = np.transpose(img, (2,1,0))

cv2.imshow('test', b)
cv2.waitKey(0)