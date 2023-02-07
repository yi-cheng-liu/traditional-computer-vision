import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb

# resd the images
img1 = cv2.imread("img1.jpg")
img2 = cv2.imread("img2.jpg")

# crop
img1 = img1[:, 504:3528]
img2 = img2[:, 504:3528]
# resize
img1 = cv2.resize(img1, (256, 256))
img2 = cv2.resize(img2, (256, 256))

# change the default BGR to RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imsave("im1.jpg",img1)
plt.imsave("im2.jpg",img2)

# change the default BGR to LAB
lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
plt.imsave("1_lab.jpg",lab1)
plt.imsave("2_lab.jpg",lab2)

# spilt the lab value and check the distance of
L1, A1, B1 = cv2.split(lab1)
plt.imsave("1_L_Channel.jpg", L1)
plt.imsave("1_A_Channel.jpg", A1)
plt.imsave("1_B_Channel.jpg", B1)
L2, A2, B2 = cv2.split(lab2)
plt.imsave("2_L_Channel.jpg", L2)
plt.imsave("2_A_Channel.jpg", A2)
plt.imsave("2_B_Channel.jpg", B2)