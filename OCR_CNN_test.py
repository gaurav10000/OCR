import numpy as np
import cv2
import pickle
import time  # Import the time module

##################################################
width = 640
height = 480
##################################################

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

# Define class names
class_names = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7", "Class_8",
               "Class_9"]


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    cv2.imshow("Processed Image", img)

    img = img.reshape(1, 32, 32, 1)

    # Predict
    classIndex = int(np.argmax(model.predict(img)))

    # Get class name from class_names list
    className = class_names[classIndex]

    print("Predicted Class:", className)

    # Introduce a delay of 1 second
    time.sleep(0.3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
