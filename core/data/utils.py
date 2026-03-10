import numpy as np
import torch
try: from torchvision.transforms import v2 as T
except ImportError: import torchvision.transforms as T
import os

VALIDATION_FRACTION = 0.2

class ImbalancedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, imb_rate=100, imb_type='exp', seed=1, validation=False, logger=None, **kwargs):
        # load base dataset
        self.base_dataset, self.logger = base_dataset, logger
        self.imb_type, self.imb_rate = imb_type, imb_rate
        self.load_base_dataset(**kwargs)

        # calculate samples_per_class
        self.get_samples_per_class(img_max=self.original_len/self.num_classes, imb_type=imb_type, imb_factor=1 / imb_rate)
        self.logger_data_info(f'--- imbalance results: {self.samples_per_class}, sum-{np.sum(self.samples_per_class)}.')

        # get samples according to samples_per_class
        self._prepare_splits(imb_type, imb_rate, seed, validation)
        self.logger_data_info(f"--- train-{len(self.data_indices)} validation-{len(self.val_indices)}.")

    def get_samples_per_class(self, img_max, imb_type='exp', imb_factor=0.01):
        cls_num = self.num_classes
        if imb_type == 'exp':
            self.samples_per_class = [int(img_max*(imb_factor**(i/(cls_num-1)))) for i in range(cls_num)]
        elif imb_type == 'step':
            half = cls_num//2
            self.samples_per_class = [int(img_max)]*half + [int(img_max*imb_factor)]*half
        else:
            self.samples_per_class = [int(img_max)]*cls_num

    def _prepare_splits(self, imb_type, imb_rate, seed, validation):
        dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split')
        os.makedirs(dir, exist_ok=True)
        split_path = os.path.join(dir, f'{self.base_dataset}-{imb_type}{imb_rate}-seed{seed}.npz')

        if os.path.exists(split_path):
            split_info = np.load(split_path)
            self.data_indices = split_info['train_indices'].tolist()
            self.val_indices = split_info['val_indices'].tolist()
        else:
            self._generate_splits(seed, split_path)

        if not validation:
            self.data_indices.extend(self.val_indices)
            self.val_indices = []

    def _generate_splits(self, seed, split_path):
        data_indices, val_indices = [], []
        indices = np.arange(self.original_len)
        rng = np.random.default_rng(seed)

        for label in range(self.num_classes):
            label_idx = indices[np.array(list(self.dataset.targets)) == label]
            rng.shuffle(label_idx)
            take_amount = self.samples_per_class[label]
            take_train = take_amount - int(take_amount * VALIDATION_FRACTION)

            data_indices.extend(label_idx[:take_train])
            val_indices.extend(label_idx[take_train: take_amount])

        self.data_indices = data_indices
        self.val_indices = val_indices
        np.savez(split_path, train_indices=data_indices, val_indices=val_indices)

    def load_base_dataset(self, **kwargs):
        raise NotImplementedError()

    def logger_data_info(self, msg):
        self.logger.log(msg) if self.logger else print(msg)

    def get_number_per_class(self, imb_type='exp', imb_factor=0.01):
        cls_num = self.num_classes
        img_max = 4500 if 'svhn' in self.base_dataset else self.original_len / cls_num
        if imb_type == 'exp':
            self.take_amount_per_label = [int(img_max * (imb_factor ** (i / (cls_num - 1)))) for i in range(cls_num)]
        elif imb_type == 'step':
            half = cls_num // 2
            self.take_amount_per_label = [int(img_max)] * half + [int(img_max * imb_factor)] * half
        else:  # uniform
            self.take_amount_per_label = [int(img_max)] * cls_num

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def original_len(self):
        return self.dataset.__len__()

    @property
    def mean_std(self):
        return self.dataset.mean_std

    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, item):
        return self.dataset[item]


def get_dataloader(dataset, shuffle=True, class_balance=False, gamma=None, num_batches=None, batch_size=256, logger=None, **kwargs):
    num_batches = int(np.ceil(len(dataset) / batch_size)) if num_batches is None else num_batches
    if class_balance:
        gamma = dataset.imb_rate / 2 if gamma is None and class_balance else gamma
        logger and logger.log(f"--- class-balanced sampler: gamma-{gamma}")
    batch_sampler = ImbalancedDataSampler(data_inds=dataset.data_indices, batch_size=batch_size, num_batches=num_batches,
                                          class_balance=class_balance, targets=dataset.targets, gamma=gamma, shuffle=shuffle)
    logger and logger.log(f"--- batch: size-{batch_size} num-{batch_sampler.__len__()}")
    logger and logger.log(f"--- {kwargs}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return dataloader


class ImbalancedDataSampler(torch.utils.data.Sampler):
    def __init__(self, data_inds, batch_size, shuffle=True, num_batches=None,
                 class_balance=False, targets=None, gamma=25.):
        super().__init__(None)
        self.batch_size, self.shuffle, self.gamma = batch_size, shuffle, gamma
        self.data_inds = data_inds

        # Class-balanced resampling
        if class_balance:
            assert targets is not None, "targets required for class balance"
            self.data_inds = self._balance_classes(data_inds, targets)

        self.num_batches = num_batches or int(np.ceil(len(self.data_inds) / batch_size))

    def _balance_classes(self, inds, targets):
        """Resample indices so each class has gamma * min_count samples."""
        num_classes = len(set(targets))
        class_inds = [[] for _ in range(num_classes)]
        for idx in inds:
            class_inds[targets[idx]].append(idx)

        min_count = min(len(c) for c in class_inds)
        balance_count = int(self.gamma * min_count)

        balanced = []
        for c_inds in class_inds:
            if len(c_inds) >= balance_count:
                balanced.extend(c_inds[: balance_count])
            else:  # oversample
                repeat, remain = divmod(balance_count, len(c_inds))
                balanced.extend(c_inds * repeat + list(np.random.choice(c_inds, remain)))
        return balanced

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            inds = np.random.permutation(self.data_inds) if self.shuffle else self.data_inds
            for i in range(0, len(inds), self.batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = list(inds[i: i + self.batch_size])
                yield batch
                batch_counter += 1

    def __len__(self): return self.num_batches