import numpy as np
import pandas as pd
from math import sqrt
def rescale(train):
    train = train.values
    feature = train[:,[0,2,4,10,11,12]]
    fea_mean = np.mean(feature, axis=0)
    fea_var = np.var(feature, axis=0)
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            feature[i,j] = (feature[i,j]-fea_mean[j])/sqrt(fea_var[j])
    label = train[:,14]
    for i in range(label.shape[0]):
        if label[i]==' <=50K':
            label[i] = -1
        else:
            label[i] = 1
    train = np.concatenate((feature, label.reshape((label.shape[0],1))), axis=1)
