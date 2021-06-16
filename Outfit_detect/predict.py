import cv2
import math
import argparse
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

faceProto="ML_models/opencv_face_detector.pbtxt"
faceModel="ML_models/opencv_face_detector_uint8.pb"
ageProto="ML_models/age_deploy.prototxt"
ageModel="ML_models/age_net.caffemodel"
genderProto="ML_models/gender_deploy.prototxt"
genderModel="ML_models/gender_net.caffemodel"
response = dict()
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

def max_no(a,b):
    if a>b:
        return a
    else:
        return b

def image_utils(img_name):
    try:
        frame = cv2.imread(f'{img_name}')
        resultImg,faceBoxes=highlightFace(faceNet,frame)
        if not faceBoxes:
            response = {"mssg": "No Face Detected!"}
            return response
        
        padding=20
        g=''
        a=''
        
        for faceBox in faceBoxes:
            a1 = max_no(0,faceBox[1]-padding)
            face=frame[max_no(0,faceBox[1]-padding):
                    min(faceBox[3]+padding,frame.shape[0]-1),max_no(0,faceBox[0]-padding)
                    :min(faceBox[2]+padding, frame.shape[1]-1)]

            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')
            g=gender
            
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')
            a=age
            
        if g=='Male':
            model = tensorflow.keras.models.load_model('ML_models/male.h5')
        else:
            model = tensorflow.keras.models.load_model('ML_models/keras_model.h5')
            
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        
        image = Image.open(f'{img_name}')
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        # image = frame.resize(size)
        #turn the image into a numpy array
        image_array = np.asarray(image)

        # display the resized image
        # image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        #print(prediction.shape)
        #print(type(prediction))
        #print(prediction)
        result=0
        max=np.amax(prediction)
        if g=='Male':
            name=['Shirt','Pant','Tshirt','Suit','Kurta','Shirt and Pant','Tshirt and pant','Tshirt and halfpant']
        else:
            name=['Skirt','Saree','Top','Jeans','Salwar Suit','Top and Skirt','Gown']
        for x in prediction:
            for y in x:
                if y == max:
                    break
                else:
                    result+=1
                    
        
        response = {"Gender": g,
                    "Age": a[1:-1],
                    "Wearing": name[result],
        }
        
        return response
    except Exception as e:
        return {"mssg": "Image Not Found"}
        
# res = image_utils('santosh.jpeg')
# print(res)