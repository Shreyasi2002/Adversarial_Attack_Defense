# Import libraries and packages
import matplotlib.pyplot as plt  # plot graphs and images
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse # for parsing arguments
import os

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

# VGG16 model
from AElib.VGG import VGG16

# Adversarial attacks
from AElib.attacks import fgsm_attack, pgd_linf

# Utilites
from AElib.utils import to_numpy_array

LABELS_MNIST = {}
LABELS_FMNIST = {}


def applyAttack(data_loader, model, eps, attack):
    accs = []
    total = 0
    correct = 0

    # epsilon = 0 means no attack
    if eps == 0:
        flag = 1

    for i, (imgs, labels) in enumerate(data_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, labels = Variable(imgs, requires_grad=True), Variable(labels)

        if attack == 'fgsm':
            adv_imgs, new_preds = fgsm_attack(model, imgs, labels, eps)
        elif attack == 'pgd':
            adv_imgs, new_preds = pgd_linf(model, imgs, labels, eps, alpha=1e-2, num_iter=40, flag=flag)
        
        # plot the perturbed image and noise
        f = plt.figure(figsize=(15, 7))
        gs = f.add_gridspec(1, 3)
        gs.update(wspace=0.5) 

        img, adv_img = imgs[0], adv_imgs[0]
        img, adv_img = to_numpy_array(img), to_numpy_array(adv_img)

        ax = f.add_subplot(gs[0, 1])
        ax.imshow(adv_img)
        ax.set_xlabel(f"Perturbed Image")

        correct += (new_preds==labels).sum().item()
        total += labels.size(0)

    accs.append((correct / total))
    print("Epsilon: {}, Test Accuracy: {}".format(eps, correct / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Adversarial Attack',
        description='''Deep Learning models are notoriously known for being overconfident in their predictions. 
        Szegedy et al. (https://arxiv.org/abs/1312.6199) discovered that Deep Neural Network models can be manipulated into making 
        wrong predictions by adding small perturbations to the input image.

        This script demonstrates how to attack a VGG16 model trained on MNIST and Fashion MNIST datasets using FGSM (Fast Gradient Sign Method) 
        and PGD (Projected Gradient Descent) attacks.
        ''',
        epilog='''The VGG16 model gives a 99.2% accuracy on the MNIST dataset and 92.1% accuracy on the Fashion MNIST dataset.''',
    )
    # inputs
    parser.add_argument('--attack', type=str, choices=['fgsm', 'pgd'], default='pgd', help='type of attack')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help='dataset to use')
    parser.add_argument('--epsilon', type=float, default=0.3,
                        help='strength of the Adversarial Attack. If FGSM attack is used, keep this value in the range [0, 1]. If PGD attack is used, keep this value in the range [0, 0.3], PGD being a stronger attack ...')

    args = vars(parser.parse_args())

    # Use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = VGG16((1,32,32), batch_norm=True)
    model.to(device)

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
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        model.load_state_dict(torch.load('./models/vgg16_mnist_model.pth'))
    elif args['dataset'] == 'fashion_mnist':
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        model.load_state_dict(torch.load('./models/vgg16_fashion-mnist_model.pth'))

    applyAttack(test_loader, model, args['epsilon'], args['attack'])


