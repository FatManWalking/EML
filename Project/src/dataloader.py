# PyTorch Datalaoder for MNIST and CIFAR10

import torch
import torchvision
import torchvision.transforms as transforms

# CIFAR10
class CIFAR10:
    def __init__(self, **kwargs):

        # batch size
        self.batch_size = kwargs["batch_size"]

        # transform
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

        self.testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        # fmt: off
        self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # fmt: on


class MNIST:
    def __init__(self, **kwargs):

        # batch size
        self.batch_size = kwargs["batch_size"]

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        self.trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=self.transform
        )
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

        self.testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=self.transform
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        # fmt: off
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
