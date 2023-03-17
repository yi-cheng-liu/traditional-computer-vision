#!/usr/bin/env python
# coding: utf-8

# # EECS 442 Homework 4: Fashion-MNIST Classification
# In this part, you will implement and train Convolutional Neural Networks (ConvNets) 
# in PyTorch to classify images. Unlike HW4, backpropagation is automatically inferred 
# by PyTorch, so you only need to write code for the forward pass.

import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split


if torch.cuda.is_available():
    print("Using the GPU. You are good to go!")
    device = 'cuda'
else:
    print("Using the CPU. Overall speed may be slowed down")
    device = 'cpu'


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        ##############################################################################
        # TODO: Design your own network, define layers here.                          #
        # Here We provide a sample of two-layer fc network from HW4 Part3.           #
        # Your solution, however, should contain convolutional layers.               #
        # Refer to PyTorch documentations of torch.nn to pick your layers.           #
        # (https://pytorch.org/docs/stable/nn.html)                                  #
        # Some common choices: Linear, Conv2d, ReLU, MaxPool2d, AvgPool2d, Dropout   #
        # If you have many layers, use nn.Sequential() to simplify your code         #
        ##############################################################################
        # from 28x28 input image to hidden layer of size 256
        self.fc1 = nn.Linear(28*28, 8) 
        # from hidden layer to 10 class scores
        self.fc2 = nn.Linear(8,10) 
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
  
    def forward(self, x):
        ##############################################################################
        # TODO: Design your own network, implement forward pass here                 # 
        ##############################################################################
        x = x.to(device)
        # Flatten each image in the batch
        x = x.view(-1,28*28) 
        x = self.fc1(x)
        # No need to define self.relu because it contains no parameters
        relu = nn.ReLU() 
        x = relu(x)
        x = self.fc2(x)
        # The loss layer will be applied outside Network class
        return x
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################


def train(model, trainloader, valloader, num_epoch=10):  # Train the model
    print("Start training...")
    trn_loss_hist = []
    trn_acc_hist = []
    val_acc_hist = []
    model.train()  # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        print('-----------------Epoch = %d-----------------' % (i+1))
        for batch, label in tqdm(trainloader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()  # Clear gradients from the previous iteration
            # This will call Network.forward() that you implement
            pred = model(batch)
            loss = criterion(pred, label)  # Calculate the loss
            running_loss.append(loss.item())
            loss.backward()  # Backprop gradients to all tensors in the network
            optimizer.step()  # Update trainable weights
        print("\n Epoch {} loss:{}".format(i+1, np.mean(running_loss)))

        # Keep track of training loss, accuracy, and validation loss
        trn_loss_hist.append(np.mean(running_loss))
        trn_acc_hist.append(evaluate(model, trainloader))
        print("\n Evaluate on validation set...")
        val_acc_hist.append(evaluate(model, valloader))
    print("Done!")
    return trn_loss_hist, trn_acc_hist, val_acc_hist


def evaluate(model, loader):  # Evaluate accuracy on validation / test set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    with torch.no_grad():  # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred, dim=1) == label).sum().item()
        acc = correct/len(loader.dataset)
        print("\n Evaluation accuracy: {}".format(acc))
        return acc


def main():
    # ## Loading Dataset
    # The dataset we use is Fashion-MNIST dataset, which is available 
    # at https://github.com/zalandoresearch/fashion-mnist and in torchvision.datasets. 
    # Fashion-MNIST has 10 classes, 60000 training+validation images (we have 
    # splitted it to have 50000 training images and 10000 validation images, 
    # but you can change the numbers), and 10000 test images.

    # Load the dataset and train, val, test splits
    print("Loading datasets...")
    # Transform from [0,255] uint8 to [0,1] float,
    # then normalize to zero mean and unit variance
    FASHION_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.2859], [0.3530]) 
                        ])
    FASHION_trainval = datasets.FashionMNIST('.', download=True, train=True,
                                            transform=FASHION_transform)
    FASHION_train = Subset(FASHION_trainval, range(50000))
    FASHION_val = Subset(FASHION_trainval, range(50000, 60000))
    FASHION_test = datasets.FashionMNIST('.', download=True, train=False,
                                        transform=FASHION_transform)
    print("Done!")

    # Create dataloaders
    ##############################################################################
    # TODO: Experiment with different batch sizes                                #
    ##############################################################################
    batch_size=32
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    trainloader = DataLoader(FASHION_train, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(FASHION_val, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(FASHION_test, batch_size=batch_size, shuffle=True)

    # ## Model
    # Initialize your model and experiment with with different optimizers, parameters 
    # (such as learning rate) and number of epochs.

    model = Network().to(device)
    criterion = nn.CrossEntropyLoss() # Specify the loss layer
    print('Your network:')
    print(summary(model, (1,28,28), device=device)) # visualize your model

    ##############################################################################
    # TODO: Modify the lines below to experiment with different optimizers,      #
    # parameters (such as learning rate) and number of epochs.                   #
    ##############################################################################
    # Set up optimization hyperparameters
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epoch = 10  # TODO: Choose an appropriate number of training epochs
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                        weight_decay=weight_decay) # Try different optimizers
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################

    # We expect you to achieve over **85%** on the test set. A valid solution that 
    # meet the requirement take no more than **10 minutes** on normal PC Intel 
    # core CPU setting. If your solution takes too long to train, try to simplify 
    # your model or reduce the number of epochs.

    start_time = time.time()
    trn_loss_hist, trn_acc_hist, val_acc_hist = train(model, trainloader,
                                                  valloader, num_epoch)
    end_time = time.time()
    print(f"Total time to train the model: {(end_time - start_time):.3f}")

    ##############################################################################
    # TODO: Note down the evaluation accuracy on test set                        #
    ##############################################################################
    print("\n Evaluate on test set")
    evaluate(model, testloader)

    ##############################################################################
    # TODO: Submit the accuracy plot                                             #
    ##############################################################################
    # visualize the training / validation accuracies
    x = np.arange(num_epoch)
    # train/val accuracies for MiniVGG
    plt.figure()
    plt.plot(x, trn_acc_hist)
    plt.plot(x, val_acc_hist)
    plt.legend(['Training', 'Validation'])
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('fashion MNIST Classification')
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('part1.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(442)
    np.random.seed(442)
    main()