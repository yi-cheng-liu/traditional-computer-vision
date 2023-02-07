import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb

if __name__ == "__main__":
    indoor = cv2.imread("indoor.png")
    outdoor = cv2.imread("outdoor.png")
    indoor_LAB = cv2.cvtColor(indoor, cv2.COLOR_RGB2Lab)
    outdoor_LAB = cv2.cvtColor(outdoor, cv2.COLOR_RGB2Lab)
    
    # Indoor RGB
    if not os.path.exists("1_indoor_RGB"):
        os.mkdir("1_indoor_RGB")
    # Outdoor RGB
    if not os.path.exists("2_outdoor_RGB"):
        os.mkdir("2_outdoor_RGB")
    # Indoor LAB
    if not os.path.exists("1_indoor_LAB"):
        os.mkdir("1_indoor_LAB")    
    # Outdoor LAB
    if not os.path.exists("1_outdoor_LAB"):
        os.mkdir("1_outdoor_LAB")
    
    for i in range(indoor.shape[-1]):
        plt.imsave("1_indoor_RGB/indoor_%d.png" % i, indoor[:,:,i])
        plt.imsave("1_indoor_RGB/indoor_gray_%d.png" % i, indoor[:,:,i], cmap='gray')
        plt.imsave("1_indoor_LAB/indoor_LAB_%d.png" % i, indoor_LAB[:,:,i])
        plt.imsave("1_indoor_LAB/indoor_LAB_gray_%d.png" % i, indoor_LAB[:,:,i], cmap='gray')
    
    for i in range(outdoor.shape[-1]):
        plt.imsave("2_outdoor_RGB/outdoor_%d.png" % i, outdoor[:,:,i])
        plt.imsave("2_outdoor_RGB/outdoor_gray_%d.png" % i, outdoor[:,:,i], cmap='gray')
        plt.imsave("1_outdoor_LAB/outdoor_LAB_%d.png" % i, outdoor_LAB[:,:,i])
        plt.imsave("1_outdoor_LAB/outdoor_LAB_gray_%d.png" % i, outdoor_LAB[:,:,i], cmap='gray')
    
    # pdb.set_trace()