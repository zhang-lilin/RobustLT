import os
import torch
import torchvision
try: from torchvision.transforms import v2 as T
except ImportError: import torchvision.transforms as T
from .cifar10lt import load_cifar10lt
from .cifar100lt import load_cifar100lt
from .tinylt import load_tinylt
from .imagenetlt import load_imagenetlt
from .utils import get_dataloader

_LOAD_DATASET_FN = {
    'cifar10lt': load_cifar10lt,
    'cifar100lt': load_cifar100lt,
    'tinylt': load_tinylt,
    'imagenetlt': load_imagenetlt,
}
DATASETS = []
for d in _LOAD_DATASET_FN:
    DATASETS.append(d)

def get_data_info(data_dir):
    """
    Returns dataset information.
    Arguments:
        data_dir (str): path to data directory.
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
    if 'cifar100' in data_dir:
        from .cifar100lt import DATA_DESC
    elif 'cifar10' in data_dir:
        from .cifar10lt import DATA_DESC
    elif 'tiny' in data_dir:
        from .tinylt import DATA_DESC
    elif 'imagenet' in data_dir:
        from .imagenetlt import DATA_DESC
    else:
        raise ValueError(f'Only data in {DATASETS} are supported!')
    return DATA_DESC


def load_imb_data(
        data_dir, logger, validation=False,
        imb_rate=100, imb_type='exp', seed=102,
        augmentation='none', num_batches=None,
        shuffle_train=True, balance_train=False, gamma=25.,
        batch_size=256, batch_size_test=256, **kwargs,
):
    dataset = os.path.basename(os.path.normpath(data_dir))
    logger and logger.log(f'Training data {dataset} (seed:{seed}, imb:{imb_type}{imb_rate})')

    load_dataset_fn = _LOAD_DATASET_FN[dataset]
    train_dataset, val_dataset = load_dataset_fn(
        data_dir=data_dir, logger=logger, validation=validation,
        imb_rate=imb_rate, imb_type=imb_type, seed=seed, augmentation=augmentation
    )
    train_dataloader = get_dataloader(
        train_dataset, class_balance=balance_train, gamma=gamma, shuffle=shuffle_train,
        num_batches=num_batches, batch_size=batch_size, logger=logger, **kwargs
    )
    if validation:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test, **kwargs)
        return train_dataset, val_dataset, train_dataloader, val_dataloader
    else:
        return train_dataset, None, train_dataloader, None


def load_test_data(data_dir, logger, batch_size_test=256, shuffle=False, **kwargs):
    dataset = os.path.basename(os.path.normpath(data_dir))
    logger and logger.log(f'Test data {dataset}')

    test_transform = T.ToTensor()
    if 'cifar100' in data_dir:
        data_dir = os.path.join(os.path.dirname(data_dir), 'cifar100')
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    elif 'cifar10' in data_dir:
        data_dir = os.path.join(os.path.dirname(data_dir), 'cifar10')
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    elif 'tiny' in data_dir:
        data_dir = os.path.join(os.path.dirname(data_dir), 'tiny-imagenet-200')
        from .tinylt import TinyImagenet
        test_dataset = TinyImagenet(root=data_dir, train=False, transform=test_transform)
    elif 'imagenet' in data_dir:
        data_dir = os.path.join(os.path.dirname(data_dir), 'imagenet')
        from .imagenetlt import Imagenet
        test_dataset = Imagenet(root=data_dir, train=False, transform=test_transform)
    else:
        raise ValueError(f'Only data in {DATASETS} are supported!')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=shuffle, **kwargs)
    logger and logger.log(f"--- length: {test_dataset.__len__()} batch-size: {batch_size_test}")

    return test_dataset, test_dataloader

