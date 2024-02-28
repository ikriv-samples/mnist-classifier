import torch
import torch.nn as nn
import pathlib

num_hidden_unites = 50
num_classes= 10 # MNIST
dropout_p = 0.5

# model
class CNN(nn.Module):
    defaultModelPath = pathlib.Path(__file__).parent.parent.resolve() / 'nist_model.pt'

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=dropout_p)
        self.fc1 = nn.Linear(320, num_hidden_unites)
        self.fc2 = nn.Linear(num_hidden_unites, num_classes)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, p=dropout_p, train=self.training)
        x = self.fc2(x)
        return x #torch.log_softmax(x, dim=1)
