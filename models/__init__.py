"""
Python module for holding our PyTorch models.
"""

def get_model(name, **model_args):
    """
    Top-level factory function for getting your models.
    """
    if name == 'cnn2d':
        from .cnn2d import CNN2D
        return CNN2D(**model_args)
    elif name == 'cnn3d':
        from .cnn3d import CNN3D
        return CNN3D(**model_args)
    elif name == 'dcgan':
        from .dcgan import get_gan
        return get_gan(**model_args)
    elif name == 'lstm':
        from .lstm import LSTM
        return LSTM(**model_args)
    elif name == 'alexnet':
        import torchvision
        return torchvision.models.alexnet(**model_args)
    elif name == 'vgg11':
        import torchvision
        return torchvision.models.vgg11(**model_args)
    elif name == 'resnet50':
        import torchvision
        return torchvision.models.resnet50(**model_args)
    elif name == 'inceptionV3':
        import torchvision
        return torchvision.models.inception_v3(**model_args)
    else:
        raise Exception('Model %s unknown' % name)
