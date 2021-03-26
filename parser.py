import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description=' EmbMatch Training')
    # Model configuration
    parser.add_argument('--dataset-path', type=str, default=os.environ.get('SEMCO_DATA_PATH', '/home/inas0003/data'),
                        help='the path to the data folder containing all datasets')
    parser.add_argument('--word-vec-path', type=str,
                        default=os.environ.get('SEMCO_WV_PATH',
                                               '/home/inas0003/data/numberbatch-en-19.08_128D.dict.pkl'),
                        help='Word vectors (Semantic Embeddings) dict path')
    parser.add_argument('--im-size', type=int, default=32,
                        help='default image size to which all images will be resized')
    parser.add_argument('--cropsize', type=int, default=32,
                        help='default cropsize to which all images will be cropped -radomly for train data and centrally for test/valid data')
    parser.add_argument('--lam-u', type=float, default=1.,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--mu', type=int, default=3,
                        help='factor of train batch size of unlabeled samples')
    parser.add_argument('--thr', type=float, default=0.95,
                        help='pseudo label threshold')
    parser.add_argument('--thr-emb', type=float, default=0.7,
                        help='pseudo label cos sim threshold for embedding path')
    parser.add_argument('--lambda-emb', type=float, default=3,
                        help='weight of embedding loss')
    parser.add_argument('--eps', type=float, default=None,
                        help='Epsilon for DBScan clustering [0-1], the less eps, the more conservative label grouping is. If left blank, will be automatically set')
    parser.add_argument('--ema-alpha', type=float, default=0.999,
                        help='decay rate for ema module')
    parser.add_argument('--parallel', type=bool, default=True,
                        help='If set to True, EmbMatch will run in parallel on all available GPUs, use CUDA_VISIBLE_DEVICES=x,y to limit parallelism to cuda:x and cuda:y only')
    parser.add_argument('--no_amp', action='store_true',
                        help='If set, Automatic mixed precision will not be used for training the model.')
    parser.add_argument('--no_progress_bar', action='store_true',
                        help='If set, progress bar will not be displayed during training')
    parser.add_argument('--cache-imgs', type=bool, default=False,
                        help='If set to True, images will be cached to memory. In case SDD is used with 4 wokers per gpu, this does not help much')
    parser.add_argument('--model_backbone', type=str, default=None,
                        help='Takes a value from [wres, resnet50, resnet18], if it is not set, the model backbone will be infered based on img size')
    parser.add_argument('--wres-k', type=int, default=2,
                        help='k parameter for wideresnet model')
    parser.add_argument('--no-imgnet-pretrained', action='store_true',
                        help='If set, the backbone model will not be pretrained using imagenet (only applicable for resnet backbones)')
    parser.add_argument('--use-pretrained', type=bool, default=False,
                        help='If set to True, the model will be initialised by the state dictionary as per the checkpoint-path argument')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='The checkpoint path to be used in case use-pretrained is set to True')
    parser.add_argument('--freeze-backbone', type=bool, default=False,
                        help='If set, only the embedding fully connected layer would be unfrozen')
    # training configuration
    parser.add_argument('--n-epoches', type=int, default=300,
                        help='number of training epoches if below is not passed')
    parser.add_argument('--break-epoch', type=int, default=None,
                        help='epoch at which training stops, use this instead of n-epoches if you want to maintain the LR scheduler')
    parser.add_argument('--early-stopping-epochs', type=int, default=0,
                        help='number of epochs after which early stopping would happen if no improvement to validation accuracy was witnessed')
    parser.add_argument('--min_wait_before_es', type=int, default=-1,
                        help='number of epochs to wait before starting to monitor best model (for saving model or for early stopping')
    parser.add_argument('--es-metric', type=str, default='accuracy',
                        help='Early stopping metric can either be accuracy or loss')
    parser.add_argument('--n-imgs-per-epoch', type=int, default=64 * 1024,
                        help='number of training images for each epoch')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for trained data')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--num-workers-per-gpu', type=int, default=4,
                        help='Number of workers per gpu')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=123,
                        help='seed for random behaviors, no seed if negative')



    ##### Launcher config #######
    parser.add_argument('--dataset-name', type=str, default='cifar100',
                        help='Name of dataset (e.g. cifar100)')
    parser.add_argument('--train-split-pickle', type=str, default='splits/cifar100_labelled_data_25_seed123.pkl',
                        help='path to pickle file with training split (generate_tst_pkls output)')
    parser.add_argument('--valid-split-pickle', type=str, default='splits/cifar100_valid_data.pkl',
                        help='path to pickle file with validation/test split (generate_tst_pkls output)')
    parser.add_argument('--classes-pickle', type=str, default='splits/cifar100_classes.pkl',
                        help='path to pickle file with classes (generate_tst_pkls output)')

    # parser.add_argument(
    args, _ = parser.parse_known_args()

    return args


if __name__ == '__main__':
    from pathlib import Path
    import pickle

    args = parse_args()
    args = dict(args._get_kwargs())
    pickle.dump(args, Path('default_args.pkl').open('wb'))