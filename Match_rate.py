import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import os

model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
saved_state_dict = torch.load('googlenet_7.pt')
model.load_state_dict(saved_state_dict)

# switch model to eval model (dropout becomes pass through)
model.eval()

# input image
# input=torch.zeros(3,224,224)

Trans = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Resize(224),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])

#label
label_sheet = pd.read_excel (r'test_and_base/base_label.xlsx')
label = label_sheet['Index'].tolist()
score_candidate = torch.load('score_candidate.pt')

target_sheet = pd.read_excel (r'test_and_base/test_label.xlsx')
target = target_sheet['Index'].tolist()



accuracy1=0
accuracy3=0
l=60

for i in range(l):
    tempdir='test_and_base/test/{}.jpg'.format(i)
    input = Trans(Image. open(tempdir))

    input=input.repeat(score_candidate.size(dim=0),1,1,1)

    with torch.no_grad():
            # send data to cuda
        if torch.cuda.is_available():
            input = input.cuda()
        score_input = model(input)


    score = score_input - score_candidate
    # distance = torch.sum(torch.abs(score),(1,2,3))
    distance = torch.sum(torch.abs(score), 1)

    order_distance = np.sort(distance.numpy())
    order = np.argsort(distance.numpy())
    order = order.astype(int)
    order = np.array(order)

    order_label= [0]*3
    for m in range(3):
        order_label[m]=label[order[m]]

    if order_label[0]==target[i]:
        accuracy1=accuracy1+1
        print('accuracy1 correct {}'.format(i))

    if order_label[0]==target[i] or order_label[1]==target[i] or order_label[2]==target[i]:
        accuracy3=accuracy3+1
        print('accuracy3 correct {}'.format(i))

accuracy3 = accuracy3/l
accuracy1 = accuracy1/l

print(accuracy1)
print(accuracy3)