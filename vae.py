import os
import pdb
import cv2

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# modified version https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, h):
        super(UnFlatten, self).__init__()
        self.h = h

    def forward(self, input):
        return input.view(input.size(0), self.h, 1, 1)

class VAE_CNN(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32):
        super(VAE_CNN, self).__init__()
        self.h = h_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(self.h),
            nn.ConvTranspose2d(h_dim, 256, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

def get_random_image(word):
    # Get image name
    image_dir = './kmnist/kkanji2/%s/' % word
    images = sorted(os.listdir(image_dir))
    sample_image = images[0]

    # Read and normalize
    sample_image = cv2.imread(image_dir + sample_image, 0)
    sample_image = (sample_image-sample_image.min())/(sample_image.max()-sample_image.min())

    # Convert to tensor
    sample_image = torch.tensor(sample_image).unsqueeze(0).unsqueeze(0)
    return sample_image.to(device).float()

def getInterpolation(latent_spaces, N): # N is the number of vectors between the spaces
    vs = []
    for index in range(len(latent_spaces) - 1):
        v1 = latent_spaces[index].squeeze()
        v2 = latent_spaces[index+1].squeeze()
        vs.append(v1.unsqueeze(0))
        for n in range(N+1):
            cur_weight = n / N
            vs.append(torch.lerp(v1, v2, cur_weight).unsqueeze(0))
        vs.append(v2.unsqueeze(0))
    return vs

def writeImages(images, video_name):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = 24
    video = cv2.VideoWriter(video_name, fourcc, fps, (64, 64))

    for idx, image in enumerate(images):
        save_image(image, './image.jpg')
        video.write(cv2.imread("./image.jpg"))

    cv2.destroyAllWindows()
    video.release()


def interpolation_experiment(model, interpolation_words, video_name):
    print(interpolation_words)
    model.eval()
    with torch.no_grad():
        ### Run the encoder so we get a list of latent spaces
        latent_spaces = []
        for word in interpolation_words:
            sample_image = get_random_image(word)
            z, mu, logvar = model.encode(sample_image)
            latent_spaces.append(z)

        ### Linearly interpolate between the images
        images = []
        interpolated_latent_spaces = getInterpolation(latent_spaces, 32)
        for new_z in interpolated_latent_spaces:
            decoded_image = model.decode(new_z).to(device).detach()
            images.append(decoded_image)

        ### Write a video given the images
        writeImages(images, video_name)

    print('Video complete!')

