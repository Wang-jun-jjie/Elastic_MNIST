import argparse
import logging
import time
from pathlib import PurePath
# select GPU on the server
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# pytorch related package 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models

import numpy as np

print('pytorch version: ' + torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser( description='PyTorch elastic transform on MNIST')
parser.add_argument('--resume', '-r',     action='store_true', help='resume from checkpoint')
parser.add_argument('--seed',             default=35328880,    type=int,   help='random seed')
parser.add_argument('--batch_size', '-b', default=8,       type=int,   help='mini-batch size (default: 8)')
parser.add_argument('--image_size',       default=28,          type=int,   help='input image size (default: 28 for MNIST)')
parser.add_argument('--alpha', '-a',      default=10,          type=int,   help='alpha for elastic transform')
parser.add_argument('--kernel_size', '-k', default=9,      type=int,   help='kernel size for elastic transform, must be odd number')
parser.add_argument('--data_directory',   default='./mnist_png',type=str, help='dataset inputs root directory')
parser.add_argument('--output_directory', default='./mnist_distort',type=str, help='dataset outputs root directory')

args = parser.parse_args()

class elastic(nn.Module):
    def __init__(self, args):
        super(elastic, self).__init__()
        self.args = args
    
    def get_tensortype(self, tensor):
        return dict(dtype=tensor.dtype, device=tensor.device)

    def get_gaussian_kernel2d(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        kernel2d = torch.mm(kernel1d[:, None], kernel1d[None, :])
        return kernel2d
    
    def gaussian_blur(self, grid, kernel_size, sigma=None):
        # kernel_size must be odd and positive integers
        if sigma is None:
            sigma = kernel_size * 0.15 + 0.35
        kernel = self.get_gaussian_kernel2d(kernel_size, sigma).to(device)
        kernel = kernel.expand(grid.shape[-3], 1, kernel.shape[0], kernel.shape[1])
        # padding = (left, right, top, bottom)
        padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
        grid = F.pad(grid, padding, mode="reflect")
        grid = F.conv2d(grid, kernel, groups=grid.shape[-3])
        return grid
    
    def get_elastic_grid(self, image_size, tensortype, alpha, kernel_size, sigma=None):
        # generate random fields
        batch_size, _, h, w = image_size
        random_fields = (torch.rand([batch_size, 2, h, w], **tensortype)*2-1)/h*2*alpha
        smooth_fields = self.gaussian_blur(random_fields, kernel_size)
        # create the grid to represent all the pixel
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, **tensortype),
                                torch.linspace(-1, 1, w, **tensortype))
        grid = torch.stack([ys, xs], -1).permute(2,1,0)
        grid = torch.stack([grid]*batch_size, 0)
        grid += smooth_fields
        # grid has shape (N, 2, H, W) here
        return grid

    def forward(self, image):
        tensortype = self.get_tensortype(image)
        grid = self.get_elastic_grid(image.size(), tensortype, self.args.alpha, self.args.kernel_size)
        grid = grid.permute(0,2,3,1)
        distort_image = F.grid_sample(image, grid, align_corners=True)
        return distort_image

# warrning: filename actually return the last dirname with it
class ImageFolderWithFilename(datasets.ImageFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithFilename, self).__getitem__(index)
        path, _ = self.imgs[index]
        path_split = PurePath(path).parts
        filename = path_split[-2] + '/' +path_split[-1]
        new_tuple = (original_tuple + (filename,))
        return new_tuple

def main():
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mean, std = (0.1307,), (0.3081,)
    print('==> Preparing dataset..')
    # Training dataset
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std), 
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])
    post_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=60, # degree
            translate=(0.1, 0.1),
            scale=(0.75, 0.95),
        ),
    ])
    train_dataset = ImageFolderWithFilename(root=args.data_directory+'/training', \
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,\
        shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
    test_dataset = ImageFolderWithFilename(root=args.data_directory+'/testing', \
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,\
        shuffle=False, drop_last=False, num_workers=8, pin_memory=True)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    model = elastic(args).to(device)
    model.eval()

    # Training set
    def training():
        # create folders
        training_directory = args.output_directory + '/training/'
        if not os.path.exists(training_directory):
            os.makedirs(training_directory)
        for classes in range(10):
            if not os.path.exists(training_directory + str(classes)):
                os.makedirs(training_directory + str(classes))
        print('==> Converting training dataset..')
        for batch_idx, (data, target, filename) in enumerate(train_loader):
            data, target, filename = data.to(device), target.to(device), filename
            output = model(data)
            output = post_transform(output)
            for index, f in enumerate(filename):
                image = output[index].detach().cpu()
                f = training_directory + f # it includes last dirname (same as target)
                torchvision.utils.save_image(image, f)
    # Testing set
    def testing():
        # create folders
        testing_directory = args.output_directory + '/testing/'
        if not os.path.exists(testing_directory):
            os.makedirs(testing_directory)
        for classes in range(10):
            if not os.path.exists(testing_directory + str(classes)):
                os.makedirs(testing_directory + str(classes))
        print('==> Converting testing dataset..')
        for batch_idx, (data, target, filename) in enumerate(test_loader):
            data, target, filename = data.to(device), target.to(device), filename
            output = model(data)
            output = post_transform(output)
            for index, f in enumerate(filename):
                image = output[index].detach().cpu()
                f = testing_directory + f # it includes last dirname (same as target)
                torchvision.utils.save_image(image, f)

    with torch.no_grad():
        training()
        testing()
        print('==> Done.')

if __name__ == "__main__":
    main()
