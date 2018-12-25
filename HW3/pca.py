import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
iris = pd.read_csv("iris.csv", header=0).values
x1 = pd.read_csv("dataI.csv", header = 0).values
x2 = pd.read_csv("dataII.csv", header = 0).values
x3 = pd.read_csv("dataIII.csv", header = 0).values
x4 = pd.read_csv("dataIV.csv", header = 0).values
x5 = pd.read_csv("dataV.csv", header = 0).values

mean = np.zeros((5,4)) # find mean of each noisy data
i = 0
for x in (x1,x2,x3,x4,x5):
    mean[i] = np.mean(x, axis=0)
    i+=1
x = [x1,x2,x3,x4,x5]
pca = PCA()
mses = np.zeros((5,5))
for d in range(5):
    data = x[d]
    for c in range(5):
        pca.fit(data) # use noisy data as training data by finding its mean and covariance
        x_hat = np.dot(pca.transform(data)[:,:c], pca.components_[:c,:]) # multiply PC's to inverse transform data
        x_hat += mean[d] # plus mean to reconstruct data
        mse = 0 # mse = mean of square of (reconstructed data - noiseless data)
        for i in range(iris.shape[0]):
            for j in range(iris.shape[1]):
                mse += (iris[i,j]-x_hat[i,j])**2
        mse = mse/(iris.shape[0])
        mses[d,c] = mse

nl_pca = PCA()
nl_mses = np.zeros((5,5))
nl_mean = np.mean(iris,axis=0) # find mean of noiseless data
for d in range(5):
    data = x[d]
    for c in range(5):
        nl_pca.fit(iris) # use noiseless data as training data by finding its mean and covariance
        nl_x_hat = np.dot(nl_pca.transform(data)[:,:c], nl_pca.components_[:c,:]) # multiply PC's to inverse transform data
        nl_x_hat += nl_mean # plus mean to reconstruct data
        nl_mse = 0 # mse = mean of square of (reconstructed data - noiseless data)
        for i in range(iris.shape[0]):
            for j in range(iris.shape[1]):
                nl_mse += (iris[i,j]-nl_x_hat[i,j])**2
        nl_mse = nl_mse/(iris.shape[0])
        nl_mses[d,c] = nl_mse

with open('chchan2-numbers.csv','wt') as f:
    f.write('0N, 1N, 2N, 3N, 4N, 0c, 1c, 2c, 3c, 4c')
    for i in range(5):
        f.write('\n'+str(nl_mses[i,0])+','+str(nl_mses[i,1])+','+str(nl_mses[i,2])+','+str(nl_mses[i,3])+','+str(nl_mses[i,4])+',')
        f.write(str(mses[i,0])+','+str(mses[i,1])+','+str(mses[i,2])+','+str(mses[i,3])+','+str(nl_mses[i,4]))

pca.fit(x2)
rc_x_hat = np.dot(pca.transform(x2)[:,:2], pca.components_[:2,:])
rc_x_hat += mean[1]
with open('chchan2-recon.csv','wt') as f:
    f.write('Sepal.Length,Sepal.Width,Petal.Length,Petal.Width')
    for i in range(iris.shape[0]):
        f.write('\n'+str(rc_x_hat[i,0])+','+str(rc_x_hat[i,1])+','+str(rc_x_hat[i,2])+','+str(rc_x_hat[i,3]))
