"""
PyTorch dataset description for a random dummy dataset.
"""

# Compatibility
from __future__ import print_function

# Externals
import torch
from torch.utils.data import TensorDataset

def make_tensor(n, shape, data_type, n_classes=None):
    if data_type == 'label':
        x = torch.randint(n_classes, [n] + shape, dtype=torch.long)
    elif data_type == 'randn':
        x = torch.randn([n] + shape)
    else:
        raise ValueError(f'Unsupported data_type {data_type}')
    return x

def make_dataset(n, input_shape, target_shape=[], input_type='randn',
                 target_type='label', n_classes=None):
    """Construct inputs and targets for one dataset and specified types.

    Currently supported input and target types are 'label' and 'randn'.
    """
    x = make_tensor(n=n, shape=input_shape, data_type=input_type, n_classes=n_classes)
    y = make_tensor(n=n, shape=target_shape, data_type=target_type, n_classes=n_classes)
    return x, y

def get_datasets(n_train, n_valid, **kwargs):
    """Construct and return random number datasets"""
    initial_seed = torch.initial_seed()
    torch.manual_seed(0)
    train_x, train_y = make_dataset(n=n_train, **kwargs)
    valid_x, valid_y = make_dataset(n=n_valid, **kwargs)
    torch.manual_seed(initial_seed & ((1<<63)-1)) # suppressing overflow error
    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)
    return train_dataset, valid_dataset

def _test():
    t, v = get_datasets()
    for d in t.tensors + v.tensors:
        print(d.size())
