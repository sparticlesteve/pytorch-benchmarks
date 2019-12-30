"""
This module defines a 3D CNN model.
"""

# Externals
import numpy as np
import torch.nn as nn

class CNN3D(nn.Module):
    """Simple 3D convolutional model"""
    def __init__(self, input_shape, output_size,
                 conv_sizes=[16, 32, 64, 128], dense_sizes=[256]):
        super(CNN3D, self).__init__()

        # Construct the convolutional layers
        conv_input_sizes = [input_shape[0]] + conv_sizes[:-1]
        cnn_layers = []
        for i in range(len(conv_sizes)):
            cnn_layers.append(nn.Conv3d(conv_input_sizes[i], conv_sizes[i],
                                        kernel_size=3, padding=1))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool3d(2))
        self.cnn = nn.Sequential(*cnn_layers)

        # Sample shape after len(conv_sizes) max-poolings
        shape = [conv_sizes[-1]] + [s // 2**len(conv_sizes) for s in input_shape[1:]]

        # Construct the fully-connected layers
        dense_input_sizes = [np.prod(shape)] + dense_sizes[:-1]
        dense_layers = []
        for i in range(len(dense_sizes)):
            dense_layers.append(nn.Linear(dense_input_sizes[i], dense_sizes[i]))
            dense_layers.append(nn.ReLU())
        dense_layers.append(nn.Linear(dense_sizes[-1], output_size))
        self.fnn = nn.Sequential(*dense_layers)

    def forward(self, x):
        h = self.cnn(x)
        h = h.view(h.size(0), -1)
        return self.fnn(h)

def get_model(**kwargs):
    return CNN3D(**kwargs)
