import torch
from dataclasses import dataclass

##############################################################################
# define noises as transformations
# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
##############################################################################
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        device = tensor.get_device()
        if device == -1:
            device = "cpu"
        return tensor + torch.normal(self.mean, self.std, tensor.size(), device=device)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class MultiplyGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        device = tensor.get_device()
        if device == -1:
            device = "cpu"
        return tensor * torch.normal(self.mean, self.std, tensor.size(), device=device)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


@dataclass
class CombinedGaussianNoise(object):
    #: Order of operations
    FirstMulThenAdd: bool = True
    #: Gauss mean value for the multiplier part
    GaussMeanMul: float = 1.0
    #: Gauss standard-deviation for the multiplier part
    GaussStdMul: float = 1.0
    #: Gauss mean value for the adder part
    GaussMeanAdd: float = 0.0
    #: Gauss standard-deviation for the adder part
    GaussStdAdd: float = 1.0

    def __call__(self, tensor):
        device = tensor.get_device()
        if device == -1:
            device = "cpu"
        if self.FirstMulThenAdd:
            out = tensor * torch.normal(
                self.GaussMeanMul, self.GaussStdMul, tensor.size(), device=device
            )
            out += torch.normal(
                self.GaussMeanAdd, self.GaussStdAdd, tensor.size(), device=device
            )
        else:
            out = tensor + torch.normal(
                self.GaussMeanAdd, self.GaussStdAdd, tensor.size(), device=device
            )
            out *= torch.normal(
                self.GaussMeanMul, self.GaussStdMul, tensor.size(), device=device
            )
        return out
