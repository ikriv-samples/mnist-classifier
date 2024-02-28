import tempfile
import torch
from torchvision import datasets, transforms

DATA_DIR = tempfile.gettempdir() + "/mldata/mnist"

batch_size = 64
image_size = (28,28)
normalization_transform = transforms.Normalize((0.1307,), (0.3081,))

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=True, download=True, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       normalization_transform
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=False)

raw_test = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=False, transform=transforms.PILToTensor()),
    batch_size=batch_size, shuffle=False)
