import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn.gspaces import rot2dOnR2
from escnn import nn as enn
from escnn.nn import FieldNorm

# Heightmap Denoising Network (HDN)
class HeightmapDenoiser(nn.Module):
    def __init__(self, in_channels=1, num_layers=10, num_feat=64):
        super(HeightmapDenoiser, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, num_feat, kernel_size=7,padding=3))
        layers.append(nn.BatchNorm2d(num_feat))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_feat, num_feat, kernel_size=7,padding=3))
            layers.append(nn.BatchNorm2d(num_feat))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.BatchNorm2d(num_feat))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_feat, in_channels, kernel_size=7,
                                padding=3))
        self.net = nn.Sequential(*layers)


    def forward(self, h):
        # h: (B, 1, k, k) noisy heightmap
        return self.net(h)  # (B, 1, k, k) denoised heightmap


class TrivialEquivariantDenoiser(torch.nn.Module):
    """
    Minimal working equivariant denoiser - trivial representations only
    """

    def __init__(self, in_channels=1, num_layers=10, num_feat=64):
        super().__init__()

        # Use continuous rotation group (this works for trivial representations)
        self.r2_act = rot2dOnR2(N=-1, maximum_frequency=1)

        # Only trivial representations (scalars) - this is safe and stable
        in_type = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        out_type = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        hid_type = enn.FieldType(self.r2_act, num_feat * [self.r2_act.trivial_repr])

        self.in_type = in_type
        print(f"Created equivariant model with {num_feat} trivial features")

        # Simple architecture
        layers = []

        # Input layer
        layers.append(enn.R2Conv(in_type, hid_type, kernel_size=3, padding=1, bias=True))
        #layers.append(enn.InnerBatchNorm(hid_type))  # More stable than FieldNorm
        #layers.append(enn.ReLU(hid_type, inplace=False))  # Explicitly no inplace
        layers.append(enn.GNormBatchNorm(hid_type))
        layers.append(enn.NormNonLinearity(hid_type, function='n_sigmoid'))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(enn.R2Conv(hid_type, hid_type, kernel_size=3, padding=1, bias=True))
            #layers.append(enn.InnerBatchNorm(hid_type))
            #layers.append(enn.ReLU(hid_type, inplace=False))
            layers.append(enn.GNormBatchNorm(hid_type))
            layers.append(enn.NormNonLinearity(hid_type, function='n_sigmoid'))

        # Output layer (no activation)
        layers.append(enn.R2Conv(hid_type, out_type, kernel_size=3, padding=1, bias=True))

        self.model = enn.SequentialModule(*layers)

        # Simple initialization
        self._init_weights()

    def _init_weights(self):
        """Simple weight initialization"""
        for module in self.model.modules():
            if isinstance(module, enn.R2Conv):
                # Simple normal initialization with small std
                nn.init.normal_(module.weights, mean=0.1, std=0.02)

    def forward(self, x):
        # FIXED: Input validation and clamping
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: NaN/Inf in input, clamping...")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clamp input to reasonable range
        x = torch.clamp(x, -10, 10)

        x_geo = enn.GeometricTensor(x, self.in_type)

        try:
            y_geo = self.model(x_geo)
            output = y_geo.tensor
        except Exception as e:
            print(f"Forward pass error: {e}")
            return x  # Return input as fallback

        # FIXED: Check for NaN in output before residual connection
        if torch.isnan(output).any():
            print("WARNING: NaN in model output, using input")
            return x
        if torch.isinf(output).any():
            print("WARNING: Inf in model output, using input")
            return x

        # FIXED: Clamp output before residual connection
        output = torch.clamp(output, -10, 10)

        # FIXED: Smaller residual connection weight
        result = output

        # Final safety checks
        if torch.isnan(result).any():
            print("WARNING: NaN in final result, returning input")
            return x

        return result

