from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import torchvision.ops as tv_nn
from typing import Any, Callable, List, Optional, Type, Union
import matplotlib.pyplot as plt

import wandb

test_accuracy = []
training_time = []


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.Identity,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, norm_layer: Optional[Callable[..., nn.Module]] = nn.Identity):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.block1_1 = BasicBlock(32, 32, 1, norm_layer)
        self.block1_2 = BasicBlock(32, 32, 1, norm_layer)
        self.block1_3 = BasicBlock(32, 32, 1, norm_layer)
        self.block2_1 = BasicBlock(32, 64, 2, norm_layer)
        self.block2_2 = BasicBlock(64, 64, 1, norm_layer)
        self.block2_3 = BasicBlock(64, 64, 1, norm_layer)
        self.block3_1 = BasicBlock(64, 128, 2, norm_layer)
        self.block3_2 = BasicBlock(128, 128, 1, norm_layer)
        self.block3_3 = BasicBlock(128, 128, 1, norm_layer)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = F.relu(x)
        x = torch.sum(x, [2, 3])
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Current time: {:.4f}; Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    time.time(),
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / data.shape[0],
                )
            )
            wandb.log({"train_loss": loss.item() / data.shape[0]})


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "Current time: {:.4f}; Test Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n".format(
            time.time(),
            epoch,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    wandb.log(
        {
            "test_loss": test_loss,
            "test_accuracy": 100.0 * correct / len(test_loader.dataset),
        }
    )
    return 100.0 * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch SVHN Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of epochs to train (default: 30)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--L2_reg", type=float, default=None, help="L2_reg (default: None)"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--activation_norm",
        type=str,
        default="batch_norm",
        help="activation norm to use",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ex5",
        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "architecture": "ResNet18",
            "dataset": "SVHN",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "seed": args.seed,
            "log_interval": args.log_interval,
            "L2_reg": args.L2_reg,
            "device": device.type,
            "activation_norm": args.activation_norm,
        },
    )

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    test_transforms = transforms.Compose([transforms.ToTensor()])
    train_transforms = [transforms.ToTensor()]
    train_transforms = transforms.Compose(train_transforms)

    dataset_train = datasets.SVHN(
        "../data", split="train", download=True, transform=train_transforms
    )
    dataset_test = datasets.SVHN(
        "../data", split="test", download=True, transform=test_transforms
    )
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    norm_layer = nn.Identity
    if args.activation_norm == "batch_norm":
        norm_layer = nn.BatchNorm2d
    elif args.activation_norm == "layer_norm":
        norm_layer = nn.LayerNorm

    model = ResNet(norm_layer=norm_layer)
    model = model.to(device)

    if args.L2_reg is not None:
        L2_reg = args.L2_reg
    else:
        L2_reg = 0.0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=L2_reg)

    print(f"Starting training at: {time.time():.4f}")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader, epoch)
        end_time = time.time()

        # Calculate test accuracy and append to list
        test_accuracy.append(test_acc)

        # Calculate training time for the epoch and append to list
        if epoch == 1:
            training_time.append(end_time - start_time)
        else:
            training_time.append(training_time[-1] + end_time - start_time)

    # Plotting test accuracy over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), test_accuracy)
    plt.title("Test Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig("test_accuracy.png")

    # Plotting test accuracy vs. training time
    plt.figure(figsize=(10, 5))
    plt.scatter(training_time, test_accuracy)
    plt.title("Test Accuracy vs. Training Time")
    plt.xlabel("Training Time (s)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.savefig("accuracy_vs_time.png")  # Save the plot as an image file


if __name__ == "__main__":
    main()
