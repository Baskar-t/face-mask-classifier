import numpy as np
import argparse
import cv2
import os
from keras_facenet import FaceNet
import pickle

# construct the argument parser and parse the argumen#ts
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")


class mask_predictor():
  def __init__(self):
    pass

  def pred_mask(image,svm_model_path):
    img=cv2.imread(image)
    #load facenet models
    print("[INFO] loading face detector model...")
    face_detect_model=FaceNet()
    #load mask classifier models
    print("[INFO] loading face mask detector model...")
    model = pickle.load(open(svm_model_path, 'rb'))
    faces_array_train = face_detect_model.extract(image, threshold=0.95)
    for j in range(len(faces_array_train)):
        face_embed_dict=faces_array_train[j]
        face_embed=face_embed_dict['embedding']
        face_confidence=face_embed_dict['confidence']
        face_box=face_embed_dict['box']
        embed_list_test.append(face_embed)
        face_box_list.append(face_box)
        face_confidence_list.append(face_confidence)
    
    face=np.array(embed_list_test)
    k=0
    label_list=[]
    proba_list=[]
    box_list=[]
    for i in face:
      i=i.reshape(1,-1)
      label_pred = model.predict(i)
      proba = model.predict_proba(i)
      if label_pred == 0:
          label = "Mask"
      else:
          label= "No Mask"
      if label=='Mask':
        proba= proba[0][0]
      else:
        proba=proba[0][1]
      box=face_box_list[k]
      box_list.append(box)
      label_list.append(label)
      proba_list.append(proba)
      k=k+1

    return img,box_list,label_list,proba_list


  

