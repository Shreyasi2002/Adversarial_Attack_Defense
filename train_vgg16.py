# Import libraries and packages

# Using PyTorch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt # plot images and graphs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse # for parsing arguments
import time
import os

# VGG16 model
from AElib.VGG import VGG16

# Let's train the model
def train(device, model, loaders, optimizer, criterion, epochs=10, save_param=True, dataset="mnist"):
    try:
        model = model.to(device)  # Load model to CUDA

        history_loss = {"train": [], "test": []}
        history_accuracy = {"train": [], "test": []}
        best_test_accuracy = 0  # variable to store the best test accuracy
        
        start_time = time.time()

        for epoch in range(epochs):
            sum_loss = {"train": 0, "test": 0}
            sum_accuracy = {"train": 0, "test": 0}

            for split in ["train", "test"]:
                if split == "train":
                    model.train()
                else:
                    model.eval()
                
                for (inputs, labels) in loaders[split]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()  # Reset gradients
                    prediction = model(inputs)
                    labels = labels.long()
                    loss = criterion(prediction, labels)
                    sum_loss[split] += loss.item()  # Update loss

                    if split == "train":
                        loss.backward()  # Compute gradients
                        optimizer.step()  # Optimize
                    
                    _,pred_label = torch.max(prediction, dim = 1)
                    pred_labels = (pred_label == labels).float()

                    batch_accuracy = pred_labels.sum().item() / inputs.size(0)
                    sum_accuracy[split] += batch_accuracy  # Update accuracy
                    
            # Compute epoch loss/accuracy
            epoch_loss = {split: sum_loss[split] / len(loaders[split]) for split in ["train", "test"]}
            epoch_accuracy = {split: sum_accuracy[split] / len(loaders[split]) for split in ["train", "test"]}

            # Store params at the best test accuracy
            if save_param and epoch_accuracy["test"] > best_test_accuracy:
                torch.save(model.state_dict(), f"./models/vgg16_{dataset}_model.pth")
                best_test_accuracy = epoch_accuracy["test"]

            # Update history
            for split in ["train", "test"]:
                history_loss[split].append(epoch_loss[split])
                history_accuracy[split].append(epoch_accuracy[split])
                
            print(f"Epoch: [{epoch + 1} | {epochs}]\nTrain Loss: {epoch_loss['train']:.4f}, Train Accuracy: {epoch_accuracy['train']:.2f}, \
            Test Loss: {epoch_loss['test']:.4f}, Test Accuracy: {epoch_accuracy['test']:.2f}, Time Taken: {(time.time() - start_time) / 60:.2f} mins")
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        # Plot loss
        plt.title("Loss")
        for split in ["train", "test"]:
            plt.plot(history_loss[split], label=split)
        plt.legend()
        plt.savefig(f"./images/vgg16_{dataset}_loss.png")
        plt.close()
        # Plot accuracy
        plt.title("Accuracy")
        for split in ["train", "test"]:
            plt.plot(history_accuracy[split], label=split)
        plt.legend()
        plt.savefig(f"./images/vgg16_{dataset}_accuracy.png")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Training VGG16 Model',
        description='''VGG16 is object detection and classification algorithm which is able to classify 1000 images of 1000 different categories with 92.7% accuracy. It is one of the popular algorithms for image classification and is easy to use with transfer learning.
        This script is used to train a VGG16 model on MNIST and Fashion MNIST datasets.
        '''
    )
    # inputs
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion-mnist'], default='mnist', help='dataset to use')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')

    args = vars(parser.parse_args())

    # Use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = VGG16((1,32,32), batch_norm=True)
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args['lr'])
    # loss function
    criterion = nn.CrossEntropyLoss()

    transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    # Create a data directory
    if not os.path.exists('./data'):
        os.makedirs('./data')
        print('Created a data directory ...')

    # Load the dataset
    if args['dataset'] == 'mnist':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif args['dataset'] == 'fashion-mnist':
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    # Define dictionary of loaders
    loaders = {"train": train_loader,
            "test": test_loader}
    
    train(device, model, loaders, optimizer, criterion, args['epochs'], dataset=args['dataset'])
