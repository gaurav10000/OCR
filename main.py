import numpy as np
import cv2
import os

#####################################################################

path = 'trainingSet'

#####################################################################
images = []
classNo = []
myList = os.listdir(path)

# print(len(myList))
noOfClasses = len(myList)

for x in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)
        classNo.append(x)
    print(x)
print(len(classNo))
