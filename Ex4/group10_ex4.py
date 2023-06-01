from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import torchvision.ops as tv_nn


class VGG11(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.layers = self._make_layers(dropout_p)

    def _make_layers(self, dropout_p):
        layers = []
        # Block 1
        layers += [
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Dropout(dropout_p),
            nn.ReLU(inplace=True),
        ]
        # MaxPool
        layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(dropout_p)]
        # Block 2
        layers += [
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Dropout(dropout_p),
            nn.ReLU(inplace=True),
        ]
        # MaxPool
        layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(dropout_p)]
        # Block 3
        layers += [
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Dropout(dropout_p),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Dropout(dropout_p),
            nn.ReLU(inplace=True),
        ]
        # MaxPool
        layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(dropout_p)]
        # Block 4
        layers += [
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.Dropout(dropout_p),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Dropout(dropout_p),
            nn.ReLU(inplace=True),
        ]
        # MaxPool
        layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(dropout_p)]
        # FC-4096
        layers += [nn.Linear(64 * 16 * 16, 4096), nn.Dropout(dropout_p)]
        layers += [nn.ReLU(inplace=True)]
        # FC-4096
        layers += [nn.Linear(4096, 4096), nn.Dropout(dropout_p)]
        layers += [nn.ReLU(inplace=True)]
        # FC-1000
        layers += [nn.Linear(4096, 1000)]  # No dropout here
        # Softmax
        layers += [nn.Softmax(dim=1)]

        return nn.ModuleList(layers)

    def forward(self, x):
        for mod in self.layers:
            x = mod(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        # L2 regularization
        if args.L2_reg:
            l2_loss = sum(
                p.norm(2) for p in model.parameters()
            )  # Calculate L2 regularization loss
            loss += (
                args.weight_decay * l2_loss
            )  # Add L2 regularization to the original loss

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
                    loss.item(),
                )
            )


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
        default=30,
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
        "--dropout_p", type=float, default=0.0, help="dropout_p (default: 0.0)"
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
        "--transform",
        type=list,
        default=[],
        metavar="t",
        help="data transform",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    test_transforms = transforms.Compose([transforms.ToTensor()])
    # Possible transformations:
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    train_transforms = [transforms.ToTensor()].extend(args.transform)
    train_transforms = transforms.Compose(train_transforms)

    dataset_train = datasets.SVHN(
        "../data", split="train", download=True, transform=train_transforms
    )
    dataset_test = datasets.SVHN(
        "../data", split="test", download=True, transform=test_transforms
    )
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = VGG11(dropout_p=args.dropout_p).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2_reg)

    print(f"Starting training at: {time.time():.4f}")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)

    if args.L2_reg is not None:
        f_name = f"trained_VGG11_L2-{args.L2_reg}.pt"
        torch.save(model.state_dict(), f_name)
        print(f"Saved model to: {f_name}")


if __name__ == "__main__":
    main()
