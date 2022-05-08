import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from PIL import Image
import os

# model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

# model=torchvision.models.alexnet(pretrained=True).features
# model=torchvision.models.vgg16(pretrained=True).features

model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
saved_state_dict = torch.load('googlenet_7.pt')
model.load_state_dict(saved_state_dict)

# switch model to eval model (dropout becomes pass through)
model.eval()

# input image
path = 'MTCNN1'
for i,j,k in os.walk(path):
    l = np.size(k)
    print(k)

candidate=torch.zeros(l,3,224,224)

Trans = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Resize(224),
        # transforms.CenterCrop(224),
        # torchvision.transforms.functional.crop(top = 0,left = 0,height = 224,width = 224),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
])

for i in range(l):
    temp=Trans(Image. open('{}/{}'.format(path,k[i])))
    candidate[i] = torchvision.transforms.functional.crop(temp, top = 0,left = 0,height = 224,width = 224)
    # candidate[i]=Trans(Image. open('{}/{}'.format(path,k[i])))

fig = plt.figure(figsize=(21, 7))
plt.subplot(1, 3, 1)
plt.imshow(candidate[0].numpy().transpose((1,2,0)))
plt.axis('off')
plt.title('Anchor')
plt.subplot(1, 3, 2)
plt.imshow(candidate[1].numpy().transpose((1,2,0)))
plt.axis('off')
plt.title('Positive')
plt.subplot(1, 3, 3)
plt.imshow(candidate[2].numpy().transpose((1,2,0)))
plt.axis('off')
plt.title('Negative')
plt.show()

score_candidate=torch.zeros(l,1000)
with torch.no_grad():
    # send data to cuda
    if torch.cuda.is_available():
        candidate = candidate.cuda()
    score_candidate = model(candidate)


    # torch.Size([1, 256, 6, 6])

torch.save(score_candidate, 'score_candidate_auto_1.pt')
print('save success')