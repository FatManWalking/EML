#### noisy operator implementations ####

# think about automatic conversion: https://pytorch.org/docs/stable/_modules/torch/ao/quantization/quantize.html#quantize


## imports
import torch.nn as nn
from torch import Tensor
from . import config as cfg
from . import impl


##############################################################################
# noise operator
##############################################################################
class NoiseOperator(nn.Module):
    def __init__(self, layer_config):
        # :)
        assert issubclass(
            type(layer_config), cfg.BaseNoiseConfig
        ), "The config must inherit from the base noise config."
        super().__init__()
        self.layer_config = layer_config
        params = self.layer_config
        # Select the noise layer implementation
        if isinstance(self.layer_config, cfg.GaussAddConfig):
            self._layer_impl = impl.AddGaussianNoise(params.GaussMean, params.GaussStd)
        elif isinstance(self.layer_config, cfg.GaussMulConfig):
            self._layer_impl = impl.MultiplyGaussianNoise(
                params.GaussMean, params.GaussStd
            )
        elif isinstance(self.layer_config, cfg.GaussCombinedConfig):
            GaussStdAdd = params.StdAmplitude * params.StdRatio
            GaussStdMul = params.StdAmplitude * (1 - params.StdRatio)
            self._layer_impl = impl.CombinedGaussianNoise(
                FirstMulThenAdd=params.FirstMulThenAdd,
                GaussMeanMul=params.GaussMeanMul,
                GaussStdMul=GaussStdMul,
                GaussMeanAdd=params.GaussMeanAdd,
                GaussStdAdd=GaussStdAdd,
            )
        elif isinstance(self.layer_config, cfg.DropoutConfig):
            # ToDo: Make sure dropout is enabled, even in eval, when it is evaluated in the NO
            # We could also just write our own implementation
            self._layer_impl = nn.Dropout(params.p, params.inplace)
        elif isinstance(self.layer_config, cfg.NoNoiseConfig):
            self._layer_impl = lambda x: x
        else:
            ValueError(f"Noise type: {self.layer_config} not supported.")

    def forward(self, x: Tensor) -> Tensor:
        # Check if the noise layer is to be executed or skipped
        if self.training and self.layer_config.enable_in_training:
            return self._layer_impl(x)
        elif (not self.training) and self.layer_config.enable_in_eval:
            return self._layer_impl(x)
        else:
            return x

    def extra_repr(self) -> str:
        """
        Add extra info to the representation string
        for better overview when printing the model features.
        """
        return "config={}, impl={}".format(
            self.layer_config,
            self._layer_impl,
        )
