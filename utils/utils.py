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