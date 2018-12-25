def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pickle
data1 = unpickle('data_batch_1')
label1 = np.array(data1[b'labels'])
feature1 = np.array(data1[b'data'])
data2 = unpickle('data_batch_2')
label2 = np.array(data2[b'labels'])
feature2 = np.array(data2[b'data'])
data3 = unpickle('data_batch_3')
label3 = np.array(data3[b'labels'])
feature3 = np.array(data3[b'data'])
data4 = unpickle('data_batch_4')
label4 = np.array(data4[b'labels'])
feature4 = np.array(data4[b'data'])
data5 = unpickle('data_batch_5')
label5 = np.array(data5[b'labels'])
feature5 = np.array(data5[b'data'])
with open('batches.meta','rb') as f:
    label_name = pickle.load(f, encoding='bytes')
name = [s.decode('ascii') for s in label_name[b'label_names']]
label = np.array([label1, label2, label3, label4, label5]) # numeric labels in training data
feature = np.array([feature1,feature2,feature3,feature4,feature5]) # features in training data
featurelist = [[] for i in range(10)] # categorize every sample into corresponding category depending on its label
for set in range(5):
    labelset = label[set]
    featureset = feature[set]
    for i in range(labelset.shape[0]):
        index = labelset[i]
        featurelist[index].append(featureset[i])
featurelist = np.array(featurelist)
featuremean = np.zeros((10,3072)) # mean images of each category
for i in range(10):
    featuremean[i] = np.mean(featurelist[i], axis=0)
pca = PCA() # PCA model in sklearn
mses = np.zeros((10))
for i in range(10):
    x = featurelist[i]
    pca.fit(x) # use training data to find its mean and covariance(mean is subtracted here)
    x_hat = np.dot(pca.transform(x)[:,:20], pca.components_[:20,:]) # use the first 20 PC's to reconstruct the mean image
    x_hat += featuremean[i] # add mean to reconstruct the mean image
    mse = mean_squared_error(x,x_hat)*3072 # calculate meam squared error(this built-in function in sklearn take mse by dividing all elements in the matrix, so we multiply it by the len of each category vector to match what the textbook suggests)
    mses[i] = mse
print(mses)
from sklearn.metrics.pairwise import euclidean_distances
D = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        D[i,j] = euclidean_distances(featuremean[i].reshape(1,-1),featuremean[j].reshape(1,-1))**2 # calculate distance matrix
A = np.identity(10)-1/10*np.ones((10,10)) # A=[I-(11^T)/N]
W = 1/2*np.matmul(np.matmul(A,D), A.T) # W=1/2(ADA^T)
import numpy.linalg as linalg
eval, evec = linalg.eig(W) # calculate eigenvalues and eigenvectors of W
eval = np.absolute(eval)
idx = eval.argsort()[::-1]
eval = eval[idx]
evec = evec[:,idx] # arrange eigenvectors in the order of descending absolute eigenvalues
U_s = evec[:,:2] # take the first 2 principle components
Sigma_s = [[np.sqrt(eval[0]),0],[0,np.sqrt(eval[1])]] # take the upperleft 2x2 submatrix of the eigenvalues
Y = np.matmul(U_s, Sigma_s) # Y=(U_s)(Sigma_s)
x = Y[:,0]
y = Y[:,1]
plt.bar(np.arange(10),mses)
plt.xlabel('numeric category')
plt.ylabel('mean squared error')
plt.show()
plt.scatter(x,y)
plt.xlabel('x-coordinates')
plt.ylabel('y-coordinates')
plt.title('mean image distance')
for i in range(10):
    plt.annotate(name[i], (x[i], y[i]), xytext = (5,5), textcoords = 'offset points')
plt.show()
