import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('car.jpg')


mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 50, 450, 290)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3,
            cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img22 = img * mask2[:, :, np.newaxis] #
# алгоритм grabCut для исходного


# plt.imshow(img22),plt.colorbar(),plt.show()


foreground = img22
foreground[np.where((img22 > [0, 0, 0]).all(axis=2))] = [255, 255, 255]


imgray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
im22, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cnt = contours[3]
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow('contours', img)
cv2.imshow('front', foreground)
cv2.waitKey()
cv2.destroyAllWindows()
