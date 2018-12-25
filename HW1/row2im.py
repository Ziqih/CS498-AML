import pandas as pd
import numpy as np
from scipy.misc import imresize
train = pd.read_csv('train.csv', header=0)
train = train.values
im = np.ravel(train[0][2:])
print(im)
im = np.reshape(im, (28,28))
x_min = (min(np.nonzero(im)[1]))
y_min = (min(np.nonzero(im)[0]))
x_max = (max(np.nonzero(im)[1]))
y_max = (max(np.nonzero(im)[0]))
im = im[y_min:y_max, x_min:x_max]
print(im.shape)
import matplotlib.pyplot as plt
im1 = imresize(im, (20,20))
# print(im1)
# plt.figure()
# plt.imshow(im1, cmap='Greys')
# plt.show()
