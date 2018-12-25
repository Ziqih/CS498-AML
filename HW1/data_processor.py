import numpy as np
import pandas as pd
from math import sqrt,pi,exp,log
from scipy.misc import imresize
def read_data(file_name):
    df = pd.read_csv(file_name, header=None)
    sam_size = df.shape[0]
    fea_size = df.shape[1]-1
    x=np.zeros((sam_size, fea_size))
    y=np.zeros((sam_size))
    for s in range(sam_size):
        y[s] = df.iloc[s,-1]
        for f in range(fea_size):
            x[s,f] = df.iloc[s,f]
    return (x,y,sam_size,fea_size)

def normdist(mean, var, x):
    prob = 1/sqrt(2*pi*var)*exp(-(x-mean)**2/(2*var))
    prob = log(prob)
    return prob

def zero_to_nan(list_name, fea_size):
    for row in range(len(list_name)):
        for col in range(fea_size):
            attr=False
            if col==2 or col==3 or col==5 or col==7:
                attr=True
            if attr and list_name[row][col]==0:
                list_name[row][col]=np.nan
    list_name = np.asarray(list_name)
    return list_name

def mean_var(arrayname):
    array_mean = np.nanmean(arrayname, axis=0)
    array_var = np.nanvar(arrayname, axis=0)
    return (array_mean, array_var)

def label_feature(data_name, sam_size, fea_size, fea_init):
    label = np.empty([sam_size,1])
    feature = np.empty([sam_size,fea_size])
    for s in range(sam_size):
        label[s]=data_name[s,fea_init-1]
        feature[s]=data_name[s,fea_init:]
    label = np.ravel(label)
    return (label, feature)

def rescale_image(image_array):
    im = image_array
    im_out = np.empty([image_array.shape[0],400])
    for i in range(image_array.shape[0]):
        im[i] = np.ravel(image_array[i])
        # print(im[i])
        im_i = np.reshape(im[i], (28,28))
        x_min = (min(np.nonzero(im_i)[1]))
        y_min = (min(np.nonzero(im_i)[0]))
        x_max = (max(np.nonzero(im_i)[1]))
        y_max = (max(np.nonzero(im_i)[0]))
        im_i = im_i[y_min:y_max, x_min:x_max]
        im_i = imresize(im_i, (20,20))
        im_out[i] = np.ravel(im_i)
    return im_out
