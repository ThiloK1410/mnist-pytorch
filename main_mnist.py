import numpy
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import gzip

from torch import optim

trainset = torchvision.datasets.MNIST(root='./source', train=True, download=True, transform=torchvision.transforms.ToTensor())
testset = torchvision.datasets.MNIST(root=".source", train=False, download=True, transform=torchvision.transforms.ToTensor())

batch_size = 32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def show_image(mat, scale: int):
    canvas = np.zeros((mat.shape[0]*scale, mat.shape[1]*scale), np.uint8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            canvas[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = mat[i, j]
    cv2.imshow("", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # size = (1, 28, 28)

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # size = (8, 14, 14)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # size = (16, 7, 7)

        self.flatten = nn.Flatten()

        # size = 16*7*7 = 784

        self.lin1 = nn.Linear(784, 100)
        self.act3 = nn.Softmax()

        self.lin2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.lin1(x)
        x = self.act3(x)

        x = self.lin2(x)

        return x


model = ConvNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_epochs = 5
for epoch in range(n_epochs):
    for inputs, labels in trainloader:
        # forward, backward, and then weight update
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = 0
    count = 0
    for inputs, labels in testloader:
        y_pred = model(inputs)
        acc += (torch.argmax(y_pred, 1) == labels).float().sum()
        count += len(labels)
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc * 100))

