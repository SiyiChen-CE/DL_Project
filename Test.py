import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


# model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

# model=torchvision.models.alexnet(pretrained=True).features
# model=torchvision.models.vgg16(pretrained=True).features

model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
saved_state_dict = torch.load('googlenet_7.pt')
model.load_state_dict(saved_state_dict)

# switch model to eval model (dropout becomes pass through)
model.eval()

test_dataset = torchvision.datasets.LFWPairs(
    './data', split='test',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    ]))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

data1,data2,target= iter(test_loader).next()

with torch.no_grad():
    for batch_idx, (data1, data2, target) in enumerate(test_loader):
        # send data to cuda
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        score1 = model(data1)
        score2 = model(data2)
        print(data1.size())
        score = score2 - score1
        # distance = torch.sum(torch.abs(score),(1,2,3))
        distance = torch.sum(torch.abs(score), 1)

threshold = 1000000
fpr, tpr,_ = metrics.roc_curve(target,(threshold-distance)/threshold)
roc_auc = metrics.auc(fpr, tpr)
np.savetxt('googlenet_7.txt', (fpr,tpr), delimiter=' ')


plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
