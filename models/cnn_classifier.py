"""
This module defines a generic CNN classifier model.
"""

# Externals
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    """
    Generic CNN classifier model with convolutions, max-pooling,
    fully connected layers, and a multi-class linear output (logits) layer.
    """
    def __init__(self, input_shape, n_classes, conv_sizes, dense_sizes, dropout=0):
        """Model constructor"""
        super(CNNClassifier, self).__init__()

        # Add the convolutional layers
        conv_layers = []
        in_size = input_shape[0]
        for conv_size in conv_sizes:
            conv_layers.append(nn.Conv2d(in_size, conv_size,
                                         kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2))
            in_size = conv_size
        self.conv_net = nn.Sequential(*conv_layers)

        # Add the dense layers
        dense_layers = []
        in_height = input_shape[1] // (2 ** len(conv_sizes))
        in_width = input_shape[2] // (2 ** len(conv_sizes))
        in_size = in_height * in_width * in_size
        for dense_size in dense_sizes:
            dense_layers.append(nn.Linear(in_size, dense_size))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout))
            in_size = dense_size
        dense_layers.append(nn.Linear(in_size, n_classes))
        self.dense_net = nn.Sequential(*dense_layers)

    def forward(self, x):
        h = self.conv_net(x)
        h = h.view(h.size(0), -1)
        return self.dense_net(h)


class CNN3D(nn.Module):
    """Simple 3D convolutional model"""
    # TODO: make more configurable
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
