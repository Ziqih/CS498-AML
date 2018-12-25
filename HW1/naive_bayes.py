import pandas as pd
import numpy as np
import random as ran
from math import log,sqrt
from data_processor import read_data, normdist,zero_to_nan, mean_var

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
for i in range(10):
    pos = [] # positive label
    neg = [] # negative label
    # classify training data based on their labels
    for row in range(test_start[i]):
        if label[row]==1:
            pos.append(fea[row])
        else:
            neg.append(fea[row])

    for row in range(test_end[i], sam_size):
        if label[row]==1:
            pos.append(fea[row])
        else:
            neg.append(fea[row])
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    # pos = zero_to_nan(pos,fea_size) # user-defined function that labels all 0 to missing data. Uncomment if necesssary
    # neg = zero_to_nan(neg,fea_size)
    (pos_mean, pos_var) = mean_var(pos) # self-defined function to find mean and variance
    (neg_mean, neg_var) = mean_var(neg)
    err = 0
    for t in range(test_start[i],test_end[i]):
        pred_label = 0 #predicting label
        prob_pos = log(pos.shape[0]/(pos.shape[0]+neg.shape[0])) #logarithmic probability of positive label
        prob_neg = log(neg.shape[0]/(pos.shape[0]+neg.shape[0])) #logarithmic probability of negative label
        for f in range(fea_size): #calculate the sum of logarithmic normal distribution probability for each class
            if (abs(pos_mean[f]-fea[t,f]) <= (3*sqrt(pos_var[f]))) or (abs(neg_mean[f]-fea[t,f]) <= (3*sqrt(neg_var[f]))):#exclude feature data greater than 3 variance to improve accuracy
                prob_pos = prob_pos+ normdist(pos_mean[f], pos_var[f], fea[t,f])
                prob_neg = prob_neg+ normdist(neg_mean[f], neg_var[f], fea[t,f])
        if prob_pos>=prob_neg: #compare the total probability and label the test data
            pred_label=1
        else:
            pred_label=0
        if pred_label!=label[t]: #if label is incorrect, error plus 1
            err = err+1
    acc[i] = (test_len-err)/test_len #accuracy = # of correct label/total label
print('accuracy is',acc)
print('average accuracy is',sum(acc)/len(acc))
