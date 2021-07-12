from datetime import datetime
import logging
import os
import sys
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import subprocess

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import re
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def interleave(x, bt):   # 1 * 64 (weakly aug labelled)  +   3 * 64 (weakly aug unlab) + 3 * 64 (strongly aug unlab)

    s = list(x.shape)  #[448, 3, 224, 224]
    res = torch.reshape(torch.transpose(x.reshape([-1, bt] + s[1:]), 1, 0), [-1] + s[1:])
    # x = x.reshape([-1, bt] + s[1:])   #torch.Size([64, 7, 3, 224, 224])
    # x = torch.transpose(x, 1,0)       # torch.Size([7, 64, 3, 224, 224])
    # x = torch.reshape(x, [-1] + s[1:])  #torch.Size([448, 3, 224, 224])
    # torch.reshape(torch.transpose(x.reshape([-1,7,3,224,244]),1,0)  , [-1,3,224,224])
    return res


def de_interleave(x, bt):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([bt, -1] + s[1:]), 1, 0), [-1] + s[1:])


def setup_default_logging(dataset_path, L,default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s"):

    dataset_name = get_dataset_name(dataset_path)
    output_dir = os.path.join(dataset_name, f'x{L}')
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(comment=f'{dataset_name}_{L}')

    logger = logging.getLogger('train')

    time_stamp = time_str()
    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'{time_stamp}_{L}_labelled_instances.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    # print
    # file_handler = logging.FileHandler()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)

    return logger, writer, time_stamp


def get_dataset_name(dataset_path):
    splitted = dataset_path.split('/')
    dataset_name = splitted[-1] if len(splitted[-1]) > 1 else splitted[-2]
    return dataset_name


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)  # return value, indices
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value

    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / (self.count + 1e-20)
        self.avg = self.sum / self.count


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.today().strftime(fmt)



def compute_stats(dataset):
    '''
    To calculate the stats of an image dataset without loading it all in memory - dataset must not be augmented so don't use the FixMatch
    dataset classes.
    Returns mean and std of the 3 channels
    '''
    loader = torch.utils.data.DataLoader(dataset,
                             batch_size=512,
                             num_workers=4,
                             shuffle=False)
    N = len(loader.dataset)
    mean = 0.0
    for images, _ in tqdm(loader, desc='mean'):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)/N

    lw = images.shape[-1] * N
    var = 0.0
    for images, _ in tqdm(loader, desc='std'):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])/lw
    std = torch.sqrt(var)
    return mean, std


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def create_dataset_structure(train_df, test_df, dataset_name, path='/home/user/data/'):
    """
    Receives two dataframes for train and test images and build the dataset directory structure as per SemCo standard
    by copying all the files into their proper path (see below schematic for directory structure).

    train_df and test_df must contain two columns: `id` containing absolute path of the raw image file ,
    `class` containing class label.
    dataset_name: the name of parent directory of the dataset (see below)
    path: the path where the dataset directory structure will be created.

    Directory structure:

    └<dataset_name>
       └───train
           │   <image1>
           │   <image2>
           │   ...
       └───test
           │   <image1-test>
           │   <image2-test>
           │   ...
       └───labels
           │   labels_train.feather
           │   labels_test.feather
    """
    assert (os.path.exists(path)), f'{path} does not exist.'
    # create directory structure and all intermediate directories
    os.makedirs(os.path.join(path, dataset_name, 'train'))
    os.makedirs(os.path.join(path, dataset_name, 'test'))
    os.makedirs(os.path.join(path, dataset_name, 'labels'))

    data = {'train': train_df.copy(), 'test': test_df.copy()}
    for split in ['train', 'test']:
        # copy files in respective folders
        df = data[split]
        for file in df.id.values:
            name = file.split('/')[-1]
            assert (re.match(r"\w*\.(?:jpg|png|jpeg)", name)), 'jpg, png or jpeg file expected'
            shutil.copyfile(file, os.path.join(path, dataset_name, split, name))

        # prepare label.feather dataframe
        df.id = df.id.apply(lambda x: x.split('/')[-1])
        df.reset_index(drop=True, inplace=True)
        df.to_feather(os.path.join(path, dataset_name, f'labels/labels_{split}.feather'))


def preprocess_stanford40(path='Stanford40'):
    """
    To preprocess Stanford40 dataset into train_df and test_df (preprocessor for create_dataset_structure)
    Assume folder Stanford40 has two subdirectories: ImageSplits and JPEGImages
    source:  http://vision.stanford.edu/Datasets/Stanford40.zip
    """
    # generating a dataframe with image id versus class
    filenames = os.listdir(os.path.join(path, 'ImageSplits/'))
    train = {'id': [], 'class': []}
    test = {'id': [], 'class': []}
    for file in filenames:
        if file in ['test.txt', 'train.txt', 'actions.txt', '.ipynb_checkpoints']:
            continue
        class_name = file.replace('_train.txt', '').replace('_test.txt', '')
        images_path = Path(os.path.join(path, 'ImageSplits', file))
        if '.txt' in str(images_path):
            image_ids = [os.path.abspath(os.path.join(path, 'JPEGImages/', elem.strip())) for elem in
                         (images_path.open('r')).readlines()]
            test_train = file.split('_')[-1]
            if test_train == 'train.txt':
                train['class'].extend([class_name] * len(image_ids))
                train['id'].extend(image_ids)
            elif test_train == 'test.txt':
                test['class'].extend([class_name] * len(image_ids))
                test['id'].extend(image_ids)
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    return train_df, test_df


def preprocess_caltech256(path='256_ObjectCategories'):
    """
    To preprocess Caltech256 dataset into train_df and test_df (preprocessor for build_lwll_data_structure)
    source : http://www.vision.caltech.edu/Image_Datasets/Caltech256/
    """
    folders = os.listdir(path)
    data = {'id': [], 'class': []}
    for folder in folders:
        class_name = re.findall(r'(?:\d{3}\.)(.+)', folder)[0].replace('-101', '')
        class_path = os.path.join(path, folder)
        for file in os.listdir(class_path):
            if not any([ext in file for ext in ['jpg', 'png', 'jpeg']]):
                continue
            data['class'].append(class_name)
            data['id'].append(os.path.abspath(os.path.join(class_path, file)))

    df = pd.DataFrame(data)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
    return train_df, test_df