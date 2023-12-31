import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

import pickle

#####################################################################

path = 'trainingSet'
testRatio = 0.2
valRatio = 0.2
imageDimension = (32, 32, 3)

batchSizeVal = 50
epochsVal = 10
stepsPerEpoch = 2000

#####################################################################

images = []
classNo = []
myList = os.listdir(path)
print("Total number of Classes detected", len(myList))
noOfClasses = len(myList)
print("Importing Classes .....")
for x in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, (imageDimension[0], imageDimension[1]))
        curImg = cv2.bitwise_not(curImg)
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")

print(" ")

images = np.array(images)
classNo = np.array(classNo)

X_train, X_test, Y_train, Y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=valRatio)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(X_train)
Y_train = to_categorical(Y_train, noOfClasses)
Y_test = to_categorical(Y_test, noOfClasses)
Y_validation = to_categorical(Y_validation, noOfClasses)


def myModel():
    noOfFilters = 60
    sizeOfFilters1 = (5, 5)
    sizeOfFilters2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500

    model = Sequential()
    model.add(Conv2D(noOfFilters, sizeOfFilters1, input_shape=(imageDimension[0], imageDimension[1], 1), activation='relu'))
    model.add(Conv2D(noOfFilters, sizeOfFilters1, activation='relu'))
    model.add(MaxPool2D(pool_size=sizeOfPool))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilters2, activation='relu'))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilters2, activation='relu'))
    model.add(MaxPool2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = myModel()

print(model.summary())

history = model.fit(dataGen.flow(X_train, Y_train, batch_size=batchSizeVal),
                    steps_per_epoch=stepsPerEpoch,
                    epochs=epochsVal,
                    validation_data=(X_validation, Y_validation),
                    shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

plt.show()

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])

pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
