import pandas as pd
import numpy as np
from scipy.misc import imsave
from data_processor import label_feature, rescale_image

train = pd.read_csv('train.csv', header=0)
train = train.values
sam_size = train.shape[0]
fea_size = train.shape[1]-2#exclude index and label
(train_label, train_fea) = label_feature(train, sam_size, fea_size, 2)
rc_train_fea = rescale_image(train_fea)
for s in range(train_fea.shape[0]):
    for f in range(train_fea.shape[1]):
        if train_fea[s,f]>=128:
            train_fea[s,f]=1
        else:
            train_fea[s,f]=0
for s in range(rc_train_fea.shape[0]):
    for f in range(rc_train_fea.shape[1]):
        if rc_train_fea[s,f]>=128:
            rc_train_fea[s,f]=1
        else:
            rc_train_fea[s,f]=0

val = pd.read_csv('val.csv', header=0)
val = val.values
sam_size = val.shape[0]
fea_size = val.shape[1]-1
(val_label, val_fea) = label_feature(val, sam_size, fea_size, 1)
rc_val_fea = rescale_image(val_fea)
for s in range(rc_val_fea.shape[0]):
    for f in range(rc_val_fea.shape[1]):
        if rc_val_fea[s,f]>=128:
            rc_val_fea[s,f]=1
        else:
            rc_val_fea[s,f]=0

test = pd.read_csv('test.csv', header=None)
test_fea = test.values
rc_test_fea = rescale_image(test_fea)
for s in range(test_fea.shape[0]):
    for f in range(test_fea.shape[1]):
        if test_fea[s,f]>=128:
            test_fea[s,f]=1
        else:
            test_fea[s,f]=0
for s in range(rc_test_fea.shape[0]):
    for f in range(rc_test_fea.shape[1]):
        if rc_test_fea[s,f]>=128:
            rc_test_fea[s,f]=1
        else:
            rc_test_fea[s,f]=0

from sklearn.naive_bayes import GaussianNB #use gaussian naive bayes in sklearn
clf = GaussianNB()
clf.fit(train_fea, train_label) #train the model using training data
test_label = clf.predict(test_fea)
test_label = np.asarray(test_label, dtype = np.int32)
with open('chchan2_1.csv','wt') as f:
    f.write('ImageId,Label')
    for i in range(test_label.shape[0]):
        f.write('\n' + str(i) + ',' + str(test_label[i]))
f.close()

plot_fea = [[] for i in range(10)]
plot_mean = np.zeros((10,784))
for i in range(test_label.shape[0]):
    curr_label = test_label[i]
    plot_fea[curr_label].append(test_fea[i])
plot_fea = np.asarray(plot_fea)
for i in range(10):
    plot_mean[i] = np.mean(plot_fea[i], axis=0)
    plot_mean_i = np.reshape(plot_mean[i],(28,28))
    imsave(name = 'unt_gaus_'+str(i)+'.jpg', arr = plot_mean_i)

clf = GaussianNB()
clf.fit(rc_train_fea, train_label)
test_label = clf.predict(rc_test_fea)
test_label = np.asarray(test_label, dtype = np.int32)
with open('chchan2_2.csv','wt') as f:
    f.write('ImageId,Label')
    for i in range(test_label.shape[0]):
        f.write('\n' + str(i) + ',' + str(test_label[i]))
f.close()

plot_fea = [[] for i in range(10)]
plot_mean = np.zeros((10,400))
for i in range(test_label.shape[0]):
    curr_label = test_label[i]
    plot_fea[curr_label].append(rc_test_fea[i])
plot_fea = np.asarray(plot_fea)
for i in range(10):
    plot_mean[i] = np.mean(plot_fea[i], axis=0)
    plot_mean_i = np.reshape(plot_mean[i],(20,20))
    imsave(name = 'str_gaus_'+str(i)+'.jpg', arr = plot_mean_i)
