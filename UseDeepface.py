from deepface import DeepFace
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import os

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]

#face verification

path = 'MTCNN1'
for i,j,label in os.walk(path):
    l = np.size(label)

distance=[0]*l
for i in range(l):
    result = DeepFace.verify(img1_path="test_and_base/test/5.jpg", img2_path="MTCNN/{}".format(label[i]),
                             model_name=models[0],enforce_detection = False)
    distance[i] = result['distance']
    print(i)

order_distance = np.sort(distance)
order = np.argsort(distance)
order = order.astype(int)
order = np.array(order)

#label
# label_sheet = pd.read_excel (r'test_and_base/base_label.xlsx')
# label = label_sheet['Index'].tolist()


print(order[0:5])
order_label= [0]*5
for i in range(5):
    order_label[i]=label[order[i]]

print(order[0:5])
fig = plt.figure()
plt.bar(range(5),order_distance[0:5],tick_label=order_label,color=['deeppink','turquoise','c','darkcyan','darkslategrey'])
plt.ylabel("Match Score")
plt.title("Match Result of Face 2 by Deepface")
plt.show()

fig.savefig('Match_result_Deepface_auto_2.png')
print('Saving image to Match_result_Deepface_auto.png')