import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
train = pd.read_csv('train.csv', header=None)
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
lambd = np.array([0.001, 0.01, 0.1, 1])#set lambda to 4 values
av = np.zeros((4,6))
bv = np.zeros((4,1))
x = np.arange(500)
for i in range(4):
    lambd_i = lambd[i]
    held_out_acc = np.zeros((500))#held out data accuracy for every 30 steps in an epoch
    mag = np.zeros((500))#Coefficient magnitude for every 30 steps in an epoch
    for epo in range(50):
        np.random.shuffle(train)#randomly shuffle all trainging data
        segment = int(train.shape[0]/10)#pick the first 10% as validation data and the rest as training data
        val_data = train[0:segment]
        val_fea = val_data[:,0:-1]
        val_label = val_data[:,-1]
        train_data = train[segment:]
        train_fea = train_data[:,0:-1]
        train_label = train_data[:,-1]
        pred_label = np.zeros(val_fea.shape[0])
        a = np.array([1,1,1,1,1,1])#assign some values to a and b
        b = 1
        held_out = train_data[0:50]#select 50 data samples as held out
        epoch = train_data[50:]
        batch_size = int(epoch.shape[0]/300)# batch size = # of training data/# of steps
        epoch = epoch[0:batch_size*300]
        batch = epoch.reshape(300, batch_size, 7)
        for step in range(300):
            step_len = 1/(0.01*step+20)# step length parameters acquired by multiple experiments
            thre = 0
            for batch_i in range(batch_size):
                data_i = batch[step, batch_i]
                thre = thre + data_i[-1]*(np.dot(a.T,data_i[0:-1])+b)# y_k(ax_k+b)
            thre = thre/batch_size
            if thre>=1:#update a if y_k(ax_k+b)>=1
                a = a-step_len*lambd_i*a
            else:#update a and b if y_k(ax_k+b)<1
                a = a-step_len*(lambd_i*a - data_i[-1]*data_i[0:-1])
                b = b+step_len*data_i[-1]
            held_out_err =0
            if step%30==0:
                index = int(step/30+j*10)
                mag[index] = np.linalg.norm(a)#the magnitude of coefficient a in every 30 seconds
                for h in range(50):
                    y = np.dot(a.T,held_out[h,0:-1])+b
                    if y>0:
                        held_out_pred = 1
                    else:
                        held_out_pred = -1
                    if held_out_pred != held_out[h,-1]:
                        held_out_err = held_out_err+1
                held_out_acc[index] = (50-held_out_err)/50#accuracy of held out in every 30 steps
        err = 0
        for t in range(val_fea.shape[0]):
            y = np.dot(a.T,val_fea[t])+b
            if y>0:
                pred_label[t] = 1
            else:
                pred_label[t] = -1
            if pred_label[t]!=val_label[t]:
                err = err+1
        acc = (val_label.shape[0]-err)/val_label.shape[0]#accuracy of predicting validation data
        print(acc)
        print('\n')
    av[i] = a
    bv[i] = b
    plt.ylim(0.0,12.0)
    plt.plot(x,mag)
    plt.legend(['lambda = 0.001', 'lambda = 0.01', 'lambda = 0.1', 'lambda = 1'], loc='upper right',fontsize = 'xx-large')

plt.xlabel('Steps',fontsize=40)
plt.ylabel('Coefficient Magnitude',fontsize=40)

test = pd.read_csv('test.csv', header=None)
test = test.values
test_fea = test[:,[0,2,4,10,11,12]]
test_fea_mean = np.mean(test_fea, axis=0)
test_fea_var = np.var(test_fea, axis=0)
for i in range(test_fea.shape[0]):
    for j in range(test_fea.shape[1]):
        test_fea[i,j] = (test_fea[i,j]-test_fea_mean[j])/sqrt(test_fea_var[j])
with open('svm1.csv','wt') as f:
    f.write('Example,Label')
    for i in range(test_fea.shape[0]):
        if np.dot(av[0].T, test_fea[i])+bv[0] > 0:
            f.write('\n' + "'" +str(i) + "'" +',' + '>50K')
        else:
            f.write('\n' + "'" +str(i) + "'" +',' + '<=50K')
with open('svm2.csv','wt') as f:
    f.write('Example,Label')
    for i in range(test_fea.shape[0]):
        if np.dot(av[1].T, test_fea[i])+bv[1] > 0:
            f.write('\n' + "'" +str(i) + "'" +',' + '>50K')
        else:
            f.write('\n' + "'" +str(i) + "'" +',' + '<=50K')
with open('svm3.csv','wt') as f:
    f.write('Example,Label')
    for i in range(test_fea.shape[0]):
        if np.dot(av[2].T, test_fea[i])+bv[2] > 0:
            f.write('\n' + "'" +str(i) + "'" +',' + '>50K')
        else:
            f.write('\n' + "'" +str(i) + "'" +',' + '<=50K')
with open('svm4.csv','wt') as f:
    f.write('Example,Label')
    for i in range(test_fea.shape[0]):
        if np.dot(av[3].T, test_fea[i])+bv[3] > 0:
            f.write('\n' + "'" +str(i) + "'" +',' + '>50K')
        else:
            f.write('\n' + "'" +str(i) + "'" +',' + '<=50K')
plt.show()
