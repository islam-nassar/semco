import numpy as np
import torch
from torch.utils.data import Dataset
import datasets.transform as T
from PIL import Image
from .randaugment import RandomAugment
from .sampler import RandomSampler, BatchSampler
import torchvision.transforms as TTV
from tqdm import tqdm
from pathlib import Path
import os


class SemCoDataset(Dataset):
    def __init__(self, dataset_path, type, size, cropsize, classes, labelled_data=None, mean=None, std=None):
        assert type in ['labelled', 'unlabelled', 'validation','test']
        super().__init__()
        self.dataset_path = dataset_path
        self.type = type
        self.size = size
        self.cropsize = cropsize
        self.classes= classes
        self.n_classes = len(classes)
        self.label_dict= {k:v for v,k in enumerate(classes)}
        self.inverse_label_dict = {v:k for v,k in enumerate(classes)}

        if mean is None or std is None:
            self.mean, self.std = 0., 1.
        else:
            self.mean = mean
            self.std = std

        if type in ['labelled', 'unlabelled']:
            assert labelled_data is not None
            if type =='labelled':
                self.data, self.label = self._load_data(labelled_data)
            elif type=='unlabelled':
                self.data = self._load_data(labelled_data)
            # common for both labelled and unlabelled
            self.trans_weak = T.Compose([
                TTV.Resize(self.size),
                TTV.Pad(padding=4, padding_mode='reflect'),
                TTV.RandomCrop(self.cropsize),
                TTV.RandomHorizontalFlip(p=0.5),
                TTV.Lambda(lambda x: np.array(x)),
                T.Normalize(self.mean, self.std),
                T.ToTensor(),
            ])
            self.trans_strong = T.Compose([
                TTV.Resize(self.size),
                TTV.Pad(padding=4, padding_mode='reflect'),
                TTV.RandomCrop(self.cropsize),
                TTV.RandomHorizontalFlip(p=0.5),
                TTV.Lambda(lambda x: np.array(x)),
                RandomAugment(2, 10),
                T.Normalize(self.mean, self.std),
                T.ToTensor(),
            ])
        elif type=='validation':
            assert labelled_data is not None
            self.data, self.label = self._load_data(labelled_data)
            self.trans = T.Compose([
                TTV.Resize(self.size),
                TTV.CenterCrop(self.cropsize),
                TTV.Lambda(lambda x: np.array(x)),
                T.Normalize(self.mean, self.std),
                T.ToTensor(),
            ])

        elif type=='test':
            self.data = self._load_data()
            self.trans = T.Compose([
                TTV.Resize(self.size),
                TTV.CenterCrop(self.cropsize),
                TTV.Lambda(lambda x: np.array(x)),
                T.Normalize(self.mean, self.std),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        img_path = self.data[idx]
        im = Image.open(img_path).convert('RGB')
        if self.type == 'labelled':
            lb = self.label[idx]
            return self.trans_weak(im), self.trans_strong(im), lb
        elif self.type == 'unlabelled':
            return self.trans_weak(im), self.trans_strong(im)
        elif self.type == 'validation':
            lb = self.label[idx]
            return self.trans(im), lb
        else:
            return self.trans(im)

    def __len__(self):
        return len(self.data)

    def _load_data(self, labelled_data=None, include_test=False):
        path = Path(self.dataset_path)
        train_path = path /'train'
        test_path = path / 'test'

        if self.type == 'labelled':
            # return only the labelled part of the data
            data_x = [(path / filename).as_posix() for filename in labelled_data.keys()]
            label_x = [self.label_dict[value] for value in labelled_data.values()]
            return data_x, label_x

        elif self.type == 'validation':
            # return only validation set
            data_v = [(path / filename).as_posix() for filename in labelled_data.keys()]
            label_v = [self.label_dict[value] for value in labelled_data.values()]
            return data_v, label_v

        elif self.type == 'test':
            # return test_features - labels are not available
            data_test = sorted((test_path / file).as_posix() for file in os.listdir(test_path))
            return data_test

        elif self.type == 'unlabelled':
            data_test = [(test_path / file).as_posix() for file in os.listdir(test_path)]
            train_filenames = [name.split('/')[1] for name in labelled_data.keys()]
            unl_train = list(set(os.listdir(train_path)) - set(train_filenames))
            unl_train = [train_path/file for file in unl_train]
            data_u = (data_test + unl_train) if include_test else unl_train

            return data_u


class SemCoDatasetRAM(SemCoDataset):

    def __getitem__(self, idx):
        im = self.data[idx]
        if self.type == 'labelled':
            lb = self.label[idx]
            return self.trans_weak(im), self.trans_strong(im), lb
        elif self.type == 'unlabelled':
            return self.trans_weak(im), self.trans_strong(im)
        elif self.type == 'validation':
            lb = self.label[idx]
            return self.trans(im), lb
        else:
            return self.trans(im)

    def _load_data(self, labelled_data=None):
        path = Path(self.dataset_path)
        train_path = path /'train'
        test_path = path / 'test'

        if self.type == 'labelled':
            # return only the labelled part of the data
            filepaths = [(path / filename).as_posix() for filename in labelled_data.keys()]
            data_x = []
            for img_path in tqdm(filepaths, desc="Loading labelled training images in RAM"):
                data_x.append(Image.open(img_path).convert('RGB'))
            label_x = [self.label_dict[value] for value in labelled_data.values()]
            return data_x, label_x

        elif self.type == 'validation':
            # return only validation set
            filepaths = [(path / filename).as_posix() for filename in labelled_data.keys()]
            data_v = []
            for img_path in tqdm(filepaths, desc="Loading validation images in RAM"):
                data_v.append(Image.open(img_path).convert('RGB'))
            label_v = [self.label_dict[value] for value in labelled_data.values()]
            return data_v, label_v

        elif self.type == 'test':
            # return test_features - labels are not available
            filepaths = sorted((test_path / file).as_posix() for file in os.listdir(test_path))
            data_test = []
            for img_path in tqdm(filepaths, desc="Loading test images in RAM"):
                data_test.append(Image.open(img_path).convert('RGB'))
            return data_test

        elif self.type == 'unlabelled':
            # data_test = [(test_path / file).as_posix() for file in os.listdir(test_path)]
            train_filenames = [name.split('/')[1] for name in labelled_data.keys()]
            unl_train = list(set(os.listdir(train_path)) - set(train_filenames))
            unl_train = [train_path/file for file in unl_train]
            data_u = []
            for img_path in tqdm(unl_train, desc="Loading unlabelled training images in RAM"):
                data_u.append(Image.open(img_path).convert('RGB'))

            return data_u


def get_train_loaders(dataset_path, classes, labelled_data, batch_size, mu, n_iters_per_epoch, size, cropsize, mean=None, std=None, num_workers=2,pin_memory=True, cache_imgs=False):
    if cache_imgs:
        ds_x = SemCoDatasetRAM(dataset_path=dataset_path, classes=classes, labelled_data=labelled_data, type='labelled',
                               size=size, cropsize=cropsize, mean=mean, std=std)
    else:
        ds_x = SemCoDataset(dataset_path= dataset_path, classes=classes, labelled_data=labelled_data, type='labelled',
                            size=size, cropsize=cropsize, mean=mean, std=std)
    sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
    dl_x = torch.utils.data.DataLoader(ds_x, batch_sampler=batch_sampler_x, num_workers=num_workers,
                                       pin_memory=pin_memory)
    if cache_imgs:
        ds_u = SemCoDatasetRAM(dataset_path= dataset_path, classes=classes, labelled_data=labelled_data, type='unlabelled',
                               size=size, cropsize=cropsize, mean=mean, std=std)
    else:
        ds_u = SemCoDataset(dataset_path=dataset_path, classes=classes, labelled_data=labelled_data,
                            type='unlabelled', size=size, cropsize=cropsize, mean=mean, std=std)
    sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
    batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
    dl_u = torch.utils.data.DataLoader(ds_u, batch_sampler=batch_sampler_u, num_workers=num_workers,
                                       pin_memory=True)
    return dl_x, dl_u


def get_val_loader(dataset_path, classes, labelled_data, batch_size, num_workers, size, cropsize, mean, std, pin_memory=True,cache_imgs=False):
    if cache_imgs:
        ds = SemCoDatasetRAM(dataset_path=dataset_path, classes=classes, labelled_data=labelled_data, type='validation',
                             size=size, cropsize=cropsize, mean=mean, std=std)
    else:
        ds = SemCoDataset(dataset_path=dataset_path, classes=classes, labelled_data=labelled_data, type='validation', size=size, cropsize=cropsize, mean=mean, std=std)
    dl = torch.utils.data.DataLoader(ds,shuffle=False,batch_size=batch_size,drop_last=False,num_workers=num_workers,pin_memory=pin_memory)
    return dl

def get_test_loader(dataset_path, classes, batch_size, num_workers, size, cropsize, mean, std, pin_memory=True, cache_imgs=False):
    if cache_imgs:
        ds = SemCoDatasetRAM(dataset_path=dataset_path, classes=classes, labelled_data=None, type='test', size=size,
                             cropsize=cropsize, mean=mean, std=std)
    else:
        ds = SemCoDataset(dataset_path=dataset_path, classes=classes, labelled_data=None, type='test', size=size, cropsize=cropsize, mean=mean, std=std)
    dl = torch.utils.data.DataLoader(ds,shuffle=False,batch_size=batch_size,drop_last=False,num_workers=num_workers,pin_memory=pin_memory)
    return dl





if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import os
    import pickle
    from pathlib import Path
    # CIFAR10_labels = 'airplane automobile bird cat deer dog frog horse ship truck'.split()
    # label_dict = {i: v for i, v in enumerate(CIFAR10_labels)}
    # dl_x, dl_u = get_train_loader('CIFAR10', 64, 7, 1024, 40, root='../data', num_workers=0)
    # batch = next(iter(dl_x))
    # for i in range(20):
    #     plt.figure()
    #     plt.imshow(batch[0][i].transpose(0, 2))
    #     plt.title(label_dict[batch[2][i].item()])
    #     plt.figure()
    #     plt.imshow(batch[1][i].transpose(0, 2))
    #     plt.title(label_dict[batch[2][i].item()])
    #     plt.show()
    # print('islam')
    dataset = "../data/lwll_datasets/external/cifar100/cifar100_full"
    classes = pickle.load(Path('../testing/cifar100_classes.pkl').open('rb'))
    valid_data = pickle.load(Path('../testing/cifar100_valid_data.pkl').open('rb'))
    labelled_data= pickle.load(Path('../testing/cifar100_labelled_data.pkl').open('rb'))

    # dl_x, dl_u = get_train_loaders(dataset, classes=classes,labelled_data=labelled_data,batch_size=8,mu=7,n_iters_per_epoch=8*1024,size=(64,64), cropsize=64,num_workers=0)
    # #
    # batch = next(iter(dl_x))
    # for i in range(8):
    #     plt.figure()
    #     plt.imshow(batch[0][i].transpose(0,2))
    #     plt.title(dl_x.dataset.inverse_label_dict[batch[2][i].item()])
    #     plt.figure()
    #     plt.imshow(batch[1][i].transpose(0, 2))
    #     plt.title(dl_x.dataset.inverse_label_dict[batch[2][i].item()])
    #     plt.show()
    # print('islam')
    # #
    # batch = next(iter(dl_u))
    # for i in range(8):
    #     plt.figure()
    #     plt.imshow(batch[0][i].transpose(0, 2))
    #     plt.figure()
    #     plt.imshow(batch[1][i].transpose(0, 2))
    #     plt.show()
    # print('islam')

    # dl_val = get_val_loader(dataset, classes, valid_data, 16, 0, (32, 32),32, 0., 1.)
    # batch = next(iter(dl_val))
    # for i in range(8):
    #     plt.figure()
    #     plt.imshow(batch[0][i].transpose(0, 2))
    #     plt.title(dl_val.dataset.inverse_label_dict[batch[1][i].item()])
    #     plt.show()
    # print('islam')

    dl_test = get_test_loader(dataset, classes,16, 0, (32, 32), 32,0., 1.)
    batch = next(iter(dl_test))
    for i in range(8):
        plt.figure()
        plt.imshow(batch[i].transpose(0, 2))
        plt.show()
    print('islam')