# -*- coding: utf-8 -*-


from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import cvlib as cv

# Load the saved model 
model =load_model('drive/My Drive/mask_model.h5')

#Image path
imageid = 'drive/My Drive/sunnywith.png'
image = cv2.imread(imageid)
if image is None:
    print("Could not read input image")
    exit()

# Detecting faces in the image
face, confidence = cv.detect_face(image)
classes = ['Withmask','Withoutmask']

# Looping through detected faces  in the image
for idx, f in enumerate(face):

    # get corner points of a face as rectangle
    (startX, startY) = f[0], f[1]
    (endX, endY) = f[2], f[3]
    # draw rectangle over face
    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

    # crop the detected face region
    face_crop = np.copy(image[startY:endY,startX:endX])

    # Preprocessing the image
    face_crop = cv2.resize(face_crop, (224,224))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    # Predicting the preprocessed image
    conf = model.predict(face_crop)[0]
    print(conf)
    print(classes)

    # get label with max probability
    idx = np.argmax(conf)
    label = classes[idx]
    start_point = (15, 15)
    end_point = (370, 80)
    thickness = -1
    Y = startY - 30 if startY - 30 > 30 else startY + 30
    if (label == 'Withmask'):
         image = cv2.rectangle(image, (startX, Y-35), (startX+160, Y+10), (0, 255, 0), thickness)
         cv2.putText(image, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if (label == 'Withoutmask'):
         image = cv2.rectangle(image, (startX, Y-35), (startX+210, Y+10), (0, 0, 255), thickness)
         cv2.putText(image, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

cv2.imwrite("drive/My Drive/detected_image.jpg",image)
