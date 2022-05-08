import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

fpr_Alex_w_feature_improve, tpr_Alex_w_feature_improve = np.loadtxt('Alex_w_feature_improve.txt',delimiter=' ')
roc_auc_Alex_w_feature_improve = metrics.auc(fpr_Alex_w_feature_improve, tpr_Alex_w_feature_improve)


fpr_VGG16_w_feature_improve, tpr_VGG16_w_feature_improve = np.loadtxt('VGG16_w_feature_improve.txt',delimiter=' ')
roc_auc_VGG16_w_feature_improve = metrics.auc(fpr_VGG16_w_feature_improve, tpr_VGG16_w_feature_improve)

fpr_Google, tpr_Google = np.loadtxt('googlenet_3.txt',delimiter=' ')
roc_auc_Google = metrics.auc(fpr_Google, tpr_Google)

fpr_Google7, tpr_Google7 = np.loadtxt('googlenet_7.txt',delimiter=' ')
roc_auc_Google7 = metrics.auc(fpr_Google7, tpr_Google7)

fig=plt.figure()
lw = 2

plt.plot(
    fpr_Alex_w_feature_improve,
    tpr_Alex_w_feature_improve,
    color="deepskyblue",
    lw=lw,
    label="AlexNet (AUC = %0.2f)" % roc_auc_Alex_w_feature_improve,
)


plt.plot(
    fpr_VGG16_w_feature_improve,
    tpr_VGG16_w_feature_improve,
    color="violet",
    lw=lw,
    label="VGG-16 (AUC = %0.2f)" % roc_auc_VGG16_w_feature_improve,
)

plt.plot(
    fpr_Google,
    tpr_Google,
    color="seagreen",
    lw=lw,
    label="GoogleNet Epoch = 4 (AUC = %0.2f)" % roc_auc_Google,
)

plt.plot(
    fpr_Google7,
    tpr_Google7,
    color="darkorange",
    lw=lw,
    label="GoogleNet Epoch = 8 (AUC = %0.2f)" % roc_auc_Google7,
)


plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve of LFW pairs Dataset")
plt.legend(loc="lower right")
plt.show()

fig.savefig('ROC_new.png')