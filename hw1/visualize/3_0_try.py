import os 
import numpy as np
import matplotlib.pyplot as plt

X = np.load("mysterydata/mysterydata.npy")

print(X)
print(X.shape)
print(X.dtype)
print(X.shape[-1])
color_channels = X.shape[-1] # 9 color channels

# create a folder for the color channels images
if not os.path.exists("0_vis"):
    os.mkdir("0_vis")
# save the color channels individually
for i in range(color_channels):
    plt.imsave("0_vis/vis_%d.png" % i, X[:,:,i])

if not os.path.exists("0_plasma"):
    os.mkdir("0_plasma")
for i in range(color_channels):
    plt.imsave("0_plasma/vis_plasma_%d.png" % i, X[:,:,i], cmap='plasma')