import os
import torch
import torchvision
from torchvision.transforms import AutoAugmentPolicy
try: from torchvision.transforms import v2 as T
except ImportError: import torchvision.transforms as T
from .utils import ImbalancedDataset

DATA_DESC = {
    'data': 'cifar100',
    'classes': tuple(range(0, 100)),
    'num_classes': 100,
    'mean': [0.5071, 0.4865, 0.4409],
    'std': [0.2673, 0.2564, 0.2762],
}

class ImbalancedCIFAR100(ImbalancedDataset):
    def load_base_dataset(self, **kwargs):
        base_dataset = DATA_DESC['data']
        assert self.base_dataset == base_dataset, f'Only imbalanced {base_dataset} is supported. Please use correct dataset!'
        self.dataset = torchvision.datasets.CIFAR100(train=True, **kwargs)
        self.dataset.num_classes = DATA_DESC['num_classes']
        self.dataset.mean_std = (DATA_DESC['mean'], DATA_DESC['std'])

def load_cifar100lt(
        data_dir, logger, validation=False,
        imb_rate=10, imb_type='exp',
        seed=1, augmentation='none'
):
    base_dataset = DATA_DESC['data']
    data_dir = os.path.join(os.path.dirname(data_dir), base_dataset)
    data_args = dict(root=data_dir, download=True)

    if augmentation == 'base':
        transform_train = T.Compose(
            [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(0.5),
             T.RandomRotation(15), T.ToTensor()])
    elif augmentation == 'aua':
        transform_aug = [T.AutoAugment(policy=AutoAugmentPolicy.CIFAR10)]
        transform_train = T.Compose(transform_aug + [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    elif augmentation == 'ra':
        transform_aug = [T.RandAugment(2,8)]
        transform_train = T.Compose(transform_aug + [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    elif augmentation == 'none':
        transform_train = T.Compose([T.ToTensor()])
    else:
        raise ValueError('Unknown augmentation {}'.format(augmentation))

    train_dataset = ImbalancedCIFAR100(
        base_dataset=base_dataset, validation=validation,
        imb_rate=imb_rate, imb_type=imb_type, seed=seed,
        transform=transform_train, logger=logger, **data_args,
    )
    if validation:
        transform_test = T.Compose([T.ToTensor()])
        val_dataset = torchvision.datasets.CIFAR100(**data_args, train=True, transform=transform_test)
        val_dataset = torch.utils.data.Subset(val_dataset, train_dataset.val_indices)
        return train_dataset, val_dataset
    return train_dataset, None