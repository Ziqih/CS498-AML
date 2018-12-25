import pandas as pd
import numpy as np
import random as ran
from data_processor import read_data
from sklearn.svm import SVC
#self-defined function that generates features and labels
(fea,label,sam_size,fea_size)=read_data('pima-indians-diabetes.csv')
### random split: pick a random number in the number of data.
### If it is within the first 80% of the data, pick 20% data after the number for testing;
### Otherwise, pick 20% data before the number for testing. Repeat the process 10 times
test_len = int(sam_size/5)
test_start = [[] for i in range(10)] #first data for testing
test_end = [[] for i in range(10)] #last data for testing
for i in range(10):
    test_start[i] = ran.randint(0,int(sam_size))
    if test_start[i] >= int(0.8*sam_size):
        test_end[i] = test_start[i]
        test_start[i] = test_end[i] - test_len
    else:
        test_end[i] = test_start[i] + test_len
acc = [None]*10
train_label = np.empty([sam_size-test_len,1])
train_fea = np.empty([sam_size-test_len,fea_size])
val_label = np.empty([test_len,1])
val_fea = np.empty([test_len,fea_size])
### pick 80% data for training and 20% data for validating
for i in range(10):
    for row in range(test_start[i]):
        train_label[row]=label[row]
        train_fea[row]=fea[row]
    for row in range(test_start[i],test_end[i]):
        val_label[row-test_start[i]]=label[row]
        val_fea[row-test_start[i]]=fea[row]
    for row in range(test_end[i], sam_size):
        train_label[row-test_len]=label[row]
        train_fea[row-test_len]=fea[row]
    train_label = np.ravel(train_label)
    val_label = np.ravel(val_label)
    clf = SVC(kernel = 'linear')
    clf.fit(train_fea, train_label)
    acc[i] = clf.score(val_fea, val_label)
    print(acc[i])
print(sum(acc)/len(acc))# find average accuracy
