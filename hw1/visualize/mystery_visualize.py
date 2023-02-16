#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb


def colormapArray(X, colors):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap. See the Bewares
    """
    N = len(colors)
    vmin = np.min(X)
    vmax = np.max(X)
    new_X = ((N - 1) * (X - vmin)) / (vmax - vmin)
    new_X = np.uint8(new_X)
    return new_X


if __name__ == "__main__":
    colors = np.load("mysterydata/colors.npy")
    data = np.load("mysterydata/mysterydata4.npy")
    
    color_print = colormapArray(data, colors)
    # store the pictures in the folder
    if not os.path.exists("3_colormapArray"):
        os.mkdir("3_colormapArray")
    for i in range(9):
        plt.imsave("3_colormapArray/colormapArray_%d.png" % i,color_print[:,:,i],cmap='plasma')
    # pdb.set_trace()
    
