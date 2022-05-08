from deepface import DeepFace
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]

#label
label_sheet = pd.read_excel (r'test_and_base/base_label.xlsx')
label = label_sheet['Index'].tolist()

target_sheet = pd.read_excel (r'test_and_base/test_label.xlsx')
target = target_sheet['Index'].tolist()

#face verification
accuracy1=0
accuracy3=0
n=60

for m in range(60):

    l=10
    distance=[0]*l
    for i in range(10):
        result = DeepFace.verify(img1_path="test_and_base/test/{}.jpg".format(m), img2_path="test_and_base/base/{}.jpg".format(i),
                             model_name=models[0],enforce_detection = False)
        distance[i] = result['distance']

    order_distance = np.sort(distance)
    order = np.argsort(distance)
    order = order.astype(int)
    order = np.array(order)


    order_label= [0]*3
    for i in range(3):
        order_label[i]=label[order[i]]

    if order_label[0]==target[m]:
        accuracy1=accuracy1+1
        print('accuracy1 correct {}'.format(m))

    if order_label[0]==target[m] or order_label[1]==target[m] or order_label[2]==target[m]:
        accuracy3=accuracy3+1
        print('accuracy3 correct {}'.format(m))

accuracy3 = accuracy3/n
accuracy1 = accuracy1/n

print(accuracy1)
print(accuracy3)