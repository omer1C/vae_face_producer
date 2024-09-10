from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
import os
import zipfile
import functions
from pathlib import Path
import time
import argparse
import requests



def show(imgs):
    # Create a grid of 8x4 images
    grid = make_grid(imgs[:32], nrow=8, padding=2, normalize=True)

    # Convert the grid tensor to numpy array and transpose to match matplotlib's format
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))

    # Plot or visualize the grid of images
    plt.figure(figsize=(20, 10))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.show()
    return None

def print_plots(num_epochs, train_losses, val_losses):
    Epochs = list(range(num_epochs))
    fig, axs = plt.subplots(1,2)
    axs[0].plot(Epochs, train_losses)
    axs[0].set_title('Training Loss')
    axs[0].set(xlabel='Epochs', ylabel='Loss')
    axs[1].plot(Epochs,val_losses)
    axs[1].set_title('Validation Loss')
    axs[1].set(xlabel='Epochs', ylabel='Loss')
    plt.show()

def vae_loss(x_recon, x, mu, logvar, logscale):
    beta = 0.1
    # scale = torch.exp(logscale)
    # dist = torch.distributions.Normal(x_recon, scale)
    # log_pxz = dist.log_prob(x)
    # BCE = log_pxz.sum(dim=(1, 2, 3))
    # BCE = nnF.mse_loss(x_recon, x)
    BCE_loss = nnF.binary_cross_entropy(x_recon, x, reduction='sum')
    # BCE_loss = nnF.mse_loss(x_recon, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    N = x.shape[0]
    loss1 = (KLD * beta + BCE_loss)
    # loss2 = (KLD * beta + BCE_loss)/N
    return loss1

def train(num_epochs, batch_size, dataset_size, model, val_break, optimizer, train_loader, test_loader):
    print('Start training!!!')
    train_losses = []
    val_losses = []
    min_loss = float('inf')
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        a1 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            imgs, _ = batch
            if torch.cuda.is_available():
                imgs = imgs.cuda()

            optimizer.zero_grad()
            # a1 = time.time()
            recon_batch, mu, logvar = model(imgs)

            # Compute VAE loss
            loss = vae_loss(recon_batch, imgs, mu, logvar, model.log_scale)
            # b1 = time.time()
            # print(f'model + loss time = {b1-a1}')
            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Since the dataset is large, train on 'dataset_size' samples.
            if dataset_size // batch_size == batch_idx:
                break
        train_losses.append(running_loss / len(train_loader.dataset))

        # Validation set
        val_loss = 0.0
        with torch.no_grad():
            model.eval()
            for batch_idx, batch in enumerate(test_loader):
                img, _ = batch
                if torch.cuda.is_available():
                    img = img.cuda()
                recon_batch, mu, logvar = model(img)
                val_loss += vae_loss(recon_batch, img, mu, logvar, model.log_scale)
                if(batch_idx >= val_break):
                   break
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        b1 = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]}, Test Loss: {val_losses[-1]}, Epoch take {b1-a1} [sec]')
        if val_loss <= min_loss:
            min_loss = val_loss
            print(f'New best epoch, number{epoch+1}, with Test Loss: {val_losses[-1]}, best model updated!')
            torch.save(model.state_dict(), "/Users/omercohen/PycharmProjects/VAEs_face_producer/" + "best_model.pth")

    return train_losses, val_losses

def plot_compare(model, train_loader):
    first_batch = next(iter(train_loader))
    imgs, _ = first_batch
    imgs = imgs[:16]
    recon_batch, mu, logvar = model(imgs)
    recon_batch = recon_batch[:16]

    show(imgs)
    show(recon_batch)

def generate_faces(model, grid_size, latent, batch_size):
    model.eval()
    dummy = torch.empty([batch_size, latent])
    z = torch.randn_like(dummy)
    # z = torch.ones_like(dummy)
    # z = torch.zeros_like(dummy)

    #insert the random noise to the decoder to create new samples.
    sample = model.decode(z)
    new_face_list = []

    j=0
    while j < grid_size:
        new_face_list.append(sample[j])
        j += 1

    grid = make_grid(new_face_list)
    show(grid)

def main_train(args):
    #Set a random seed:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    #Creating a path to the data:
    mid_path = os.getcwd()
    data_path = os.path.join(mid_path, 'datasets/')
    IMAGE_SIZE = 64
    #Download celebA data set :
    trfm = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)), transforms.ToTensor()])

    training_data = datasets.CelebA(root=data_path, split='train', download=True, transform=trfm)
    test_data = datasets.CelebA(root=data_path, split='test', download=False, transform=trfm)


    #Define variabels:
    learning_rate = args.learning_rate
    batch_size = 128
    num_epochs = args.epochs
    dataset_size = 30000
    latent1 = args.latent
    val_break = int((dataset_size/10)//batch_size + 1)
    #VAE Class inputs:
    enc_in_chnl = 3
    enc_num_hidden = 32
    dec_in_chnl = 256
    dec_num_hidden = 256
    ###
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    ###
    if torch.cuda.is_available():
        model_1 = functions.VAE(enc_in_chnl, enc_num_hidden, dec_in_chnl, dec_num_hidden, latent1).cuda()
        model_1.weight_init(mean=0, std=0.02)
    else:
        model_1 = functions.VAE(enc_in_chnl, enc_num_hidden, dec_in_chnl, dec_num_hidden, latent1)
        model_1.weight_init(mean=0, std=0.02)


    # model_1.load_state_dict(torch.load('/Users/omercohen/PycharmProjects/VAEs_face_producer/VAEsbest_model.pth', map_location=torch.device('cpu')))
    optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

    train_losses, val_losses = train(num_epochs, batch_size, dataset_size, model_1, val_break, optimizer=optimizer,
                                     train_loader=train_loader, test_loader=test_loader)
    print_plots(num_epochs, train_losses, val_losses)
    generate_faces(model_1, grid_size=16, latent=latent1, batch_size=batch_size)


def main(args):
    #Generate case:
    if args.action == "Generate":
        model_url = "https://huggingface.co/omer1C/VAEsbest_model.pth/resolve/main/VAEsbest_model.pth"

        # Download the file
        print("Download weights...")
        response = requests.get(model_url)
        # Save the file locally
        with open('VAEsbest_model.pth', 'wb') as f:
            f.write(response.content)
        print("Weights downloaded successfully!")

        #Initial the model:
        #VAE Class inputs:
        enc_in_chnl = 3; enc_num_hidden = 32 ; dec_in_chnl = 256 ;dec_num_hidden = 256 ;latent1 = 256
        model_1 = functions.VAE(enc_in_chnl, enc_num_hidden, dec_in_chnl, dec_num_hidden, latent1)
        model_1.weight_init(mean=0, std=0.02)
        # Upload the weights :
        weights_path = os.getcwd()
        weights_path = os.path.join(weights_path,'VAEsbest_model.pth/' )

        model_1.load_state_dict(torch.load(weights_path,
                                           map_location=torch.device('cpu')))
        print("Generate faces...")
        batch_size = 128
        generate_faces(model_1, grid_size=16, latent=latent1, batch_size=batch_size)
        create = True
        while create :
            user_gen_ans = str(input('Keep generate? Y/N'))
            if user_gen_ans == 'Y' :
                generate_faces(model_1, grid_size=16, latent=latent1, batch_size=batch_size)
            else:
                create = False
                print('Thank you for visiting 🤗')
    # Train case
    elif args.action == 'Train':
        main_train(args)

if __name__ =="__main__":
    """Main function for face producer with VAEs model."""

    parser = argparse.ArgumentParser(description="Train and generate new human faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # General settings.
    parser.add_argument("--action", type=str, default="Generate",
                        help="Choose action, gene")
    parser.add_argument("--epochs", type=str, default="20",
                        help="Choose action, gene")
    parser.add_argument("--latent", type=str, default="256",
                        help="Choose action, gene")
    parser.add_argument("--learning_rate", type=str, default="5e-3",
                        help="Choose action, gene")

