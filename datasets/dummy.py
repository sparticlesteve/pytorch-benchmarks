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
        return torch.randint(n_classes, [n] + shape, dtype=torch.long)
    elif data_type == 'randn':
        return torch.randn([n] + shape)
    else:
        raise ValueError(f'Unsupported data_type {data_type}')

def make_dataset(n, input_shape, target_shape=[], input_type='randn',
                 target_type='label', n_classes=None):
    """Construct inputs and targets for one dataset and specified types.

    Currently supported input and target types are 'label' and 'randn'.
    """
    x = make_tensor(n=n, shape=input_shape, data_type=input_type, n_classes=n_classes)
    if target_shape is not None:
        y = make_tensor(n=n, shape=target_shape, data_type=target_type, n_classes=n_classes)
        return TensorDataset(x, y)
    return TensorDataset(x)

def get_datasets(n_train, n_valid, **kwargs):
    """Construct and return random number datasets"""
    initial_seed = torch.initial_seed()
    torch.manual_seed(0)
    train_dataset = make_dataset(n=n_train, **kwargs)
    valid_dataset = make_dataset(n=n_valid, **kwargs)
    torch.manual_seed(initial_seed & ((1<<63)-1)) # suppressing overflow error
    return train_dataset, valid_dataset

def _test():
    t, v = get_datasets()
    for d in t.tensors + v.tensors:
        print(d.size())
