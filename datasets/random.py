"""
This module contains a PyTorch random synthetic dataset implementation.
"""

import torch

def _make_tensor(shape, data_type, n_classes=None):
    if data_type == 'label':
        return torch.randint(n_classes, shape, dtype=torch.long)
    elif data_type == 'randn':
        return torch.randn(shape)
    else:
        raise ValueError(f'Unsupported data_type {data_type}')

class PregeneratedRandomDataset(torch.utils.data.Dataset):
    """Random number synthetic dataset.

    Pre-generates a specified number of samples to draw from.
    """

    def __init__(self, n, input_shape, target_shape=[], input_type='randn',
                 target_type='label', n_classes=None, n_gen=1024):
        self.n = n
        x = _make_tensor(shape=[n_gen] + input_shape,
                         data_type=input_type, n_classes=n_classes)
        if target_shape is None:
            self.data = torch.utils.data.TensorDataset(x)
        else:
            y = _make_tensor(shape=[n_gen] + target_shape,
                             data_type=target_type, n_classes=n_classes)
            self.data = torch.utils.data.TensorDataset(x, y)

    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.data[index % len(self.data)]

def get_datasets(n_train, n_valid, **kwargs):
    """Construct and return random number datasets"""
    #initial_seed = torch.initial_seed()
    #torch.manual_seed(0)
    train_dataset = PregeneratedRandomDataset(n=n_train, **kwargs)
    valid_dataset = PregeneratedRandomDataset(n=n_valid, **kwargs)
    #torch.manual_seed(initial_seed & ((1<<63)-1)) # suppressing overflow error
    return train_dataset, valid_dataset
