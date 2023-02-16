import os 
import numpy as np
import matplotlib.pyplot as plt

x = np.load("./mysterydata/mysterydata2.npy")

y1 = np.log1p(x[:,:,0])
y2 = np.log1p(x[:,:,1])
y3 = np.dstack((y1,y2))

if not os.path.exists("1_log1p"):
    os.mkdir("1_log1p")
for i in range(9):
    plt.imsave("1_log1p/wide_range_nolog_%d.png" % i,x[:,:,i])        # no log
    plt.imsave("1_log1p/wide_range_log_%d.png" % i, np.log1p(x[:,:,i]))  # log

channels = x.shape[-1]

print(y1.shape)
print(y2.shape)
print(y3.shape)

