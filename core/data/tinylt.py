import os
import shutil
import urllib
import zipfile
import torch
import os.path
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from torchvision.transforms import AutoAugmentPolicy
from tqdm import tqdm
try: from torchvision.transforms import v2 as T
except ImportError: import torchvision.transforms as T
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import ImageFolder
from .utils import ImbalancedDataset


NUM_CLASSES = 20
DATA_DESC = {
    'data': f'tiny-imagenet-200',
    'classes': tuple(range(0, NUM_CLASSES)),
    'num_classes': NUM_CLASSES,
    'mean': [0.4802, 0.4481, 0.3975], 
    'std': [0.2302, 0.2265, 0.2262],
}


class TinyImagenet(VisionDataset):
    """ TinyImagenet Dataset.
    Note: We download TinyImagenet dataset from <http://cs231n.stanford.edu/tiny-imagenet-200.zip>, then repack it as `.pt` format.

    Args:
        root (string): Root directory of the dataset where the data is stored.
        split (string): One of {'train', 'val'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``T.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    splits = {
        "train": 'train.pt',
        "val": 'val.pt',
    }
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(
            self,
            root: str = '../dataset_data/tiny-imagenet-200/',
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download=True,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        os.makedirs(root, exist_ok=True)

        self.download_dir = os.path.join(self.root, 'raw_data')
        os.makedirs(self.download_dir, exist_ok=True)
        self.prepare()

        split = 'train' if train else 'val'
        fpath = os.path.join(root, self.splits[split])
        data = torch.load(fpath, weights_only=False)
        self.data, self.targets = data['data'], data["targets"]

        num_classes = DATA_DESC['num_classes']
        if num_classes < 200:
            selected_indices = [i for i, label in enumerate(self.targets) if label < num_classes]
            self.data = self.data[selected_indices]
            self.targets = np.array([self.targets[i] for i in selected_indices])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self):
        zip_path = os.path.join(self.download_dir, 'tiny-imagenet-200.zip')
        with urllib.request.urlopen(self.url) as response:
            total_size = int(response.getheader('Content-Length'))

        with tqdm(total=total_size, unit='B', unit_scale=True, desc=zip_path) as pbar:
            def reporthook(count, block_size, total_size):
                pbar.update(block_size)

            urllib.request.urlretrieve(self.url, zip_path, reporthook=reporthook)

    def extract(self) -> None:
        zip_path = os.path.join(self.download_dir, 'tiny-imagenet-200.zip')
        dataset_path = os.path.join(self.download_dir, 'tiny-imagenet-200')
        if not os.path.exists(zip_path):
            print("Downloading TinyImageNet...")
            self.download()
        else:
            print("Zip file already exists, skip downloading.")

        val_dir = os.path.join(dataset_path, "val")
        val_img_dir = os.path.join(val_dir, "images")
        val_anno_file = os.path.join(val_dir, "val_annotations.txt")

        if not os.path.exists(val_dir):
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.download_dir)
        else:
            print("Dataset already extracted.")

        with open(val_anno_file, "r") as f:
            lines = f.readlines()
        anno_dict = {line.split("\t")[0]: line.split("\t")[1] for line in lines}

        for img, cls in anno_dict.items():
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(val_img_dir, img)
            dst = os.path.join(cls_dir, img)
            if os.path.exists(src):
                shutil.move(src, dst)

        if os.path.exists(val_img_dir):
            shutil.rmtree(val_img_dir)

        print("TinyImageNet ready at:", dataset_path)

    def prepare(self):
        dataset_path = os.path.join(self.download_dir, 'tiny-imagenet-200')
        if os.path.exists(os.path.join(dataset_path, "val", "images")):
            self.extract()

        def dataset_to_numpy(dataset):
            data, targets = [], []

            for img, target in dataset:
                arr = (img.numpy() * 255).astype(np.uint8)  # [C,H,W], 0-255
                arr = np.transpose(arr, (1, 2, 0))  # [H,W,C]
                data.append(arr)
                targets.append(int(target))

            return np.array(data), np.array(targets)

        transform = T.Compose([T.ToTensor()])
        for key in self.splits:
            data_path = os.path.join(self.root, f"{key}.pt")
            if os.path.exists(data_path):
                continue

            dataset = ImageFolder(root=os.path.join(dataset_path, key), transform=transform)
            print(f"Converting {key} set...")
            data, targets = dataset_to_numpy(dataset)

            torch.save({"data": data, "targets": targets.tolist()}, data_path)

        print(f"Saved prepared dataset at {self.root}")

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

class ImbalancedTinyImagenet(ImbalancedDataset):
    def load_base_dataset(self, **kwargs):
        base_dataset = DATA_DESC['data']
        assert self.base_dataset == base_dataset, f'Only imbalanced {base_dataset} is supported. Please use correct dataset!'
        self.dataset = TinyImagenet(train=True, **kwargs)
        self.dataset.num_classes = DATA_DESC['num_classes']
        self.dataset.mean_std = (DATA_DESC['mean'], DATA_DESC['std'])

def load_tinylt(
        data_dir, logger, validation=False,
        imb_rate=50, imb_type='exp', seed=1,
        augmentation='none',
):
    base_dataset = DATA_DESC['data']
    data_dir = os.path.join(os.path.dirname(data_dir), base_dataset)
    data_args = dict(root=data_dir, download=True)

    if augmentation == 'base':
        transform_train = T.Compose([
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()]
        )
    elif augmentation == 'aua':
        transform_aug = [T.AutoAugment(policy=AutoAugmentPolicy.CIFAR10)]
        transform_train = T.Compose(transform_aug + [
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    elif augmentation == 'ra':
        transform_aug = [T.RandAugment(2,8)]
        transform_train = T.Compose(transform_aug + [
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    elif augmentation == 'none':
        transform_train = T.Compose([T.ToTensor()])
    else:
        raise ValueError('Unknown augmentation {}'.format(augmentation))

    train_dataset = ImbalancedTinyImagenet(
        base_dataset=base_dataset, validation=validation,
        imb_rate=imb_rate, imb_type=imb_type, seed=seed,
        transform=transform_train, logger=logger, **data_args,
    )
    if validation:
        transform_test = T.Compose([T.ToTensor()])
        val_dataset = TinyImagenet(**data_args, train=True, transform=transform_test)
        val_dataset = torch.utils.data.Subset(val_dataset, train_dataset.val_indices)
        return train_dataset, val_dataset
    return train_dataset, None


