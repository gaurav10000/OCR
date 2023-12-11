import cv2
import numpy as np



def preProcessing(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.equalizeHist(img) # makes the light of the image distributed
    img = img/255
    return img




img = cv2.imread("./trainingSet/0/img_1.jpg")
img = np.asarray(img)
img = cv2.resize(img,(320,320))
# img = preProcessing(img)

img = cv2.bitwise_not(img)
cv2.imshow("image", img)



cv2.waitKey(0)


