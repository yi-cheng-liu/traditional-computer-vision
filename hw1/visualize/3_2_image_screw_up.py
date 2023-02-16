import os 
import numpy as np
import matplotlib.pyplot as plt

X = np.load("mysterydata/mysterydata3.npy")
    
# Determine the range of the data
vmin = np.nanmin(X)
vmax = np.nanmax(X)

X[np.isnan(X)] = vmin

if not os.path.exists("2_vis"):
    os.mkdir("2_vis")
for i in range(9):
    plt.imsave("2_vis/screw_up_%d.png" % i,X[:,:,i], vmin=vmin, vmax=vmax)