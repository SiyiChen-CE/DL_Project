import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
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
        # torchvision.transforms.Resize(224),
        # torchvision.transforms.Resize([284, 376]),
        torchvision.transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])


tempdir='test_images_face/2.jpg'
tempdir='test_and_base/test/5.jpg'
# tempdir='test_and_base/test/37.jpg'
print(np.size(Image. open(tempdir)))
input = Trans(Image. open(tempdir).convert('RGB'))
score_candidate = torch.load('score_candidate_auto.pt')

fig = plt.figure(figsize=(21, 7))
plt.subplot(1, 3, 1)
plt.imshow(input.numpy().transpose((1,2,0)))
plt.axis('off')
plt.title('Anchor')

input=input.repeat(score_candidate.size(dim=0),1,1,1)


#label
# label_sheet = pd.read_excel (r'test_and_base/base_label.xlsx')
# label = label_sheet['Index'].tolist()

path = 'MTCNN'
for i,j,label in os.walk(path):
    l = np.size(label)
    print(label)

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

order_label= [0]*5
for i in range(5):
    order_label[i]=label[order[i]]

print(order[0:5])
fig = plt.figure()
ax=plt.subplot()
plt.bar(range(5),order_distance[0:5],tick_label=order_label,color=['deeppink','turquoise','c','darkcyan','darkslategrey'])
plt.ylabel("Match Score")
plt.title("Match Result of Face 2 by GoogleNet")
# ax.set_ylim(bottom = 300)
plt.show()

fig.savefig('Match_result_GoogleNet_auto_2.png')
print('Saving image to Match_result_GoogleNet_auto.png')