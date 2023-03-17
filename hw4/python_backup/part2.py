#!/usr/bin/env python
# coding: utf-8

# # EECS 442 Homework 4 - PyTorch ConvNets
# In this notebook we will explore how to use a pre-trained PyTorch convolution neural
# network (ConvNet).

import os
import json
import torch
import torchvision
import torchvision.transforms as T
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
SQUEEZENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
SQUEEZENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)
from PIL import Image

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

if torch.cuda.is_available():
    print("Using the GPU. You are good to go!")
    device = 'cuda'
else:
    print("Using the CPU. Overall speed may be slowed down")
    device = 'cpu'


def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
              std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled
  
def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

def main():
    # For all of our experiments, we will start with a convolutional neural network 
    # which was pretrained to perform image classification on ImageNet [1]. We can 
    # use any model here, but for the purposes of this assignment we will use 
    # SqueezeNet [2], which achieves accuracies comparable to AlexNet but with a 
    # significantly reduced parameter count and computational complexity.
    # 
    # Using SqueezeNet rather than AlexNet or VGG or ResNet means that we can easily
    # perform all experiments without heavy computation. Run the following cell to 
    # download and initialize your model.
    # 
    # [1] Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, 
    # Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, 
    # Alexander C. Berg and Li Fei-Fei. ImageNet Large Scale Visual Recognition 
    # Challenge. IJCV, 2015
    # 
    # [2] Iandola et al, "SqueezeNet: AlexNet-level accuracy with 50x fewer 
    # parameters and < 0.5MB model size", arXiv 2016
    print('Download and load the pretrained SqueezeNet model.')
    model = torchvision.models.squeezenet1_1(pretrained=True).to(device)

    # We don't want to train the model, so tell PyTorch not to compute gradients
    # with respect to model parameters.
    for param in model.parameters():
        param.requires_grad = False
        
    # Make sure the model is in "eval" mode
    model.eval()

    # you may see warning regarding initialization deprecated, that's fine, 
    # please continue to next steps

    # Loading the imagenet class labels
    # Download the dataset with following command line if you haven't done it yet
    # wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json

    class_idx = json.load(open("imagenet_class_index.json"))
    idx2label = {k:class_idx[str(k)][1] for k in range(len(class_idx))}



    ###### Task 6 - Pre-trained Convolution Network ######
    # In order to get a better sense of the classification decisions made by 
    # convolutional networks, your job is now to experiment by running whatever
    # images you want through a model pretrained on ImageNet. These can be images
    # from your own photo collection, from the internet, or somewhere else but 
    # they **should belong to one of the ImageNet classes**. Look at the `idx2label`
    #  dictionary for all the ImageNetclasses.
    # 
    # You need to find:
    # 1. One image (`img1`) where the SqueezeNet model gives reasonable predictions,
    # and produces a category label that seems to correctly describe the content 
    # of the image
    # 2. One image (`img2`) where the SqueezeNet model gives unreasonable 
    # predictions, and produces a category label that does not correctly describe
    # the content of the image.
    # 
    ###############################################################################
    # TODO: Upload your image and run the forward pass to get the ImageNet class. #
    # This code will crash when you run it, since the maxresdefault.jpg image is  #
    # not found. You should upload your own images to the Colab notebook and edit #
    # these lines to load your own image.                                         #
    ###############################################################################
    img1 = None
    img2 = None
    names = ['image1 name', 'image2 name']
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    for i, img in enumerate([img1, img2]):
        X = preprocess(img).to(device)
        pred_class = torch.argmax(model(X)).item()
        plt.figure(figsize=(6,8))
        plt.imshow(img)
        plt.title('Predicted Class: %s' % idx2label[pred_class])
        plt.axis('off')
        plt.savefig(f'{names[i]}_pred.jpg')
        plt.show()
        plt.close()
        plt.cla()

if __name__ == '__main__':
    main()
