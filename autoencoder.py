# Import libraries and packages

# Using PyTorch
import torch
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt # plot images and graphs
import random
import time
import argparse # for parsing arguments
import os
from sys import argv

# VGG16 model
from AElib.VGG import VGG16

# Adversarial attacks
from AElib.attacks import fgsm_attack, pgd_linf

# Utilites
from AElib.utils import to_numpy_array, load_checkpoint, save_checkpoint

# Autoencoder model
from AElib.autoencoder import ConvAutoencoder_GELU, EarlyStopping


# Parameters
BATCH_SIZE = 64

# For FGSM attack
EPS_FGSM = 0.6

# For PGD attack
EPS_PGD = 0.3
ALPHA = 0.01
NUM_ITER = 40

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

convAE = ConvAutoencoder_GELU(device)
convAE.to(device)

# Loss function
criterion = nn.MSELoss()
# Optimizer
optimizer = optim.Adam(convAE.parameters(), lr=0.001)

early_stopping = EarlyStopping()

# Get random indexes to display the images
def get_random_index(dataset):
    idx1 = random.randint(0, len(dataset))
    idx2 = random.randint(0, len(dataset))
    idx3 = random.randint(0, len(dataset))
    return idx1, idx2, idx3

def get_pretrained_model(dataset, attack):
    file_path = ''
    if dataset == 'mnist':
        if attack == 'fgsm':
            file_path = './models/ae_mnist_fgsm.pth.tar'
        else:
            file_path = './models/ae_mnist_pgd.pth.tar'
    else:
        if attack == 'fgsm':
            file_path = './models/ae_fmnist_fgsm.pth.tar'
        else:
            file_path = './models/ae_fmnist_pgd.pth.tar'
    return file_path


def train(model, num_epochs, attack_type, train_loader, val_loader, dataset, load_model=False, eps_fgsm=EPS_FGSM, eps_pgd=EPS_PGD):
    train_losses, val_losses = [], []
    file_path = get_pretrained_model(dataset, attack_type)

    if load_model:
        train_losses, val_losses = load_checkpoint(torch.load(file_path), convAE, optimizer)
        print("Loaded pre-trained model ...")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss, val_loss = 0.0, 0.0
        train_count, val_count = 0, 0
        for i, (imgs, labels) in enumerate(train_loader):
            batch_size = imgs.shape[0]
            imgs, labels = Variable(imgs.to(device), requires_grad=True), Variable(labels.to(device))

            if attack_type == 'fgsm':
                adv_imgs, _ = fgsm_attack(model, imgs, labels, eps_fgsm)
            else:
                adv_imgs, _ = pgd_linf(model, imgs, labels, eps_pgd, ALPHA, NUM_ITER)
            
            train_count += len(adv_imgs)

            adv_imgs = adv_imgs.to(device)

            optimizer.zero_grad()
            rec_imgs = convAE.forward(adv_imgs)
            loss = criterion(imgs, rec_imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size

        for _, (imgs, labels) in enumerate(val_loader):
            batch_size = imgs.shape[0]
            imgs, labels = Variable(imgs.to(device), requires_grad=True), Variable(labels.to(device))
            if attack_type == 'fgsm':
                adv_imgs, _ = fgsm_attack(model, imgs, labels, eps_fgsm)
            else:
                adv_imgs, _ = pgd_linf(model, imgs, labels, eps_pgd, ALPHA, NUM_ITER)
            
            val_count += len(adv_imgs)

            adv_imgs = adv_imgs.to(device)

            optimizer.zero_grad()
            rec_imgs = convAE.forward(adv_imgs)
            loss = criterion(imgs, rec_imgs)
            loss.backward()
            optimizer.step()
            val_loss += loss.item() * batch_size

        train_loss, val_loss = train_loss / train_count, val_loss / val_count

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch: {epoch+1} / {num_epochs}, Train_loss: {train_loss:.4f}, Val_loss: {val_loss:.4f}, \
              Time_taken: {(time.time()-start_time)/60:.2f} mins')

        checkpoint = {
            'convAE_state_dict': convAE.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses, 'val_losses': val_losses
        }

        save_checkpoint(checkpoint, file_path)
        early_stopping(train_loss, val_loss)
        if early_stopping.early_stop:
            print("Early Stopping critieria satisfied")
            break
            
    # Plot the training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('number of epochs')
    plt.legend()

    plt.savefig(f'./images/loss_{attack_type}_{dataset}.png', bbox_inches='tight', dpi=300)
    plt.close()


# Test the model
def test(model, test_loader, attack_type, dataset, eps_fgsm=EPS_FGSM, eps_pgd=EPS_PGD):
    file_path = get_pretrained_model(dataset, attack_type)
    convAE.load_state_dict(torch.load(file_path)['convAE_state_dict'])
    
    correct, total = 0, 0
    
    tot_time = 0

    for _, (imgs, labels) in enumerate(test_loader):
        batch_size = imgs.shape[0]
        imgs, labels = Variable(imgs.to(device), requires_grad=True), Variable(labels.to(device))

        if attack_type == 'fgsm':
            adv_imgs, _ = fgsm_attack(model, imgs, labels, eps_fgsm)
        else:
            adv_imgs, _ = pgd_linf(model, imgs, labels, eps_pgd, ALPHA, NUM_ITER)

        adv_imgs = adv_imgs.to(device)
        
        time_comp = time.time()

        with torch.no_grad():
            rec_imgs = convAE(adv_imgs)

        y_preds = model(rec_imgs).argmax(dim=1)
        correct += (y_preds == labels).sum().item()
        total += labels.size(0)
        
        tot_time += (time.time() - time_comp)

    print("Test Accuracy: {}".format(correct / total))
    print(f'Time taken for defense against a single instance of attack: {(tot_time)/10000:.4f} sec')
    print(f'Total time taken: {tot_time:.4f} sec')

# Visualise the reconstructed image
def visualise(model, dataset, val_loader, attack_type, eps_fgsm=EPS_FGSM, eps_pgd=EPS_PGD):
    count = 0
    idx1, idx2, idx3 = get_random_index(val_loader)
    for i, (imgs, labels) in enumerate(val_loader):
        if i in [idx1, idx2, idx3]:
            count += 1
            batch_size = imgs.shape[0]
            imgs, labels = Variable(imgs.to(device), requires_grad=True), Variable(labels.to(device))

            if attack_type == 'fgsm':
                adv_imgs, _ = fgsm_attack(model, imgs, labels, eps_fgsm)
            else:
                adv_imgs, _ = pgd_linf(model, imgs, labels, eps_pgd, ALPHA, NUM_ITER)

            adv_imgs = adv_imgs.to(device)

            with torch.no_grad():
                rec_imgs = convAE(adv_imgs)

            imgs, adv_imgs, rec_imgs = to_numpy_array(imgs[0]), to_numpy_array(adv_imgs[0]), to_numpy_array(rec_imgs[0])

            f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
            ax1.imshow(adv_imgs)
            ax1.set_title("Adversarial Image")
            ax2.imshow(rec_imgs)
            ax2.set_title("Recreated Image")
            ax3.imshow(imgs)
            ax3.set_title("Original Image")

            f.savefig(f'./images/reconstruction_{attack_type}_{dataset}_{count}.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Adversarial Defense using Autoencoders',
        description='''This work is based on enhancing the robustness of targeted classifier 
        models against adversarial attacks. To achieve this, a convolutional autoencoder-based 
        approach is employed that effectively counters adversarial perturbations introduced to 
        the input images.
        '''
    )
    # inputs
    parser.add_argument('--attack', type=str, choices=['fgsm', 'pgd'], default='pgd', help='type of attack')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion-mnist'], default='mnist', help='dataset to use')
    parser.add_argument('--action', type=str, choices=['train', 'test'], default='train', help='train the model or test the model')
    parser.add_argument('--use_pretrained', type=bool, default=True, help='use pretrained model (True/False) (set this to True while testing the model)')
    parser.add_argument('--epsilon', type=float, default=0.3, required=('train' in argv),
                        help='strength of the Adversarial Attack while training. If FGSM attack is used, keep this value in the range [0, 1]. If PGD attack is used, keep this value in the range [0, 0.3], PGD being a stronger attack ...')
    parser.add_argument('--epochs', type=int, default=10, required=('train' in argv), help='number of epochs to train the model')

    args = vars(parser.parse_args())

    model = VGG16((1,32,32), batch_norm=True)
    model.to(device)
    model.load_state_dict(torch.load(f'./models/vgg16_{args["dataset"]}_model.pth'))

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
        dataset = datasets.MNIST(root= './data', train = True, download =True, transform = transform)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [50000, 10000])

        test_dataset = datasets.MNIST(root= './data', train = False, download =True, transform = transform)
    elif args['dataset'] == 'fashion-mnist':
        dataset = datasets.FashionMNIST(root= './data', train = True, download =True, transform = transform)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [50000, 10000])

        test_dataset = datasets.FashionMNIST(root= './data', train = False, download =True, transform = transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if args['action'] == 'train':
        print('Training the model ...\n')
        train(model, args['epochs'], args['attack'], 
              train_loader, val_loader, args['dataset'], load_model=args['use_pretrained'], 
              eps_fgsm=args['epsilon'], eps_pgd=args['epsilon'])
    else:
        if not args['use_pretrained']:
            print('Please set use_pretrained to True to test the model ...')
            print('Exiting ...')
            exit()
        print('Testing the model ...\n')
        test(model, test_loader, args['attack'], args['dataset'], eps_fgsm=args['epsilon'], 
             eps_pgd=args['epsilon'])
    
    visualise(model, args['dataset'], val_loader, args['attack'], eps_fgsm=args['epsilon'], eps_pgd=args['epsilon'])



