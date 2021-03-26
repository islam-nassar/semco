from model.semco import SemCo
import parser as parser
import pickle
from pathlib import Path
import torch
import os

STATS = {'imagenet':((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         'mini_imagenet':((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         'cifar100':((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
         'cifar10': ((0.4913, 0.4821, 0.4465), (0.247, 0.2434, 0.2615)),
         'domain_net-real': ((0.6059, 0.5890, 0.5558), (0.3195, 0.3128, 0.3352)),
        'mnist': ((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
        'fashion_mnist': ((0.286, 0.286, 0.286), (0.353, 0.353, 0.353)),
         }

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()  # wrapper to argparse but using same names for argparse lib
    dataset_name = args.dataset_name
    if args.valid_split_pickle is None and args.classes_pickle is None:
        classes = pickle.load(Path(f'splits/{dataset_name}_classes.pkl').open('rb'))
        valid_data = pickle.load(Path(f'splits/{dataset_name}_valid_data.pkl').open('rb'))
        setattr(args, 'classes_pickle', f'splits/{dataset_name}_classes.pkl')
        setattr(args, 'valid_split_pickle', f'splits/{dataset_name}_valid_data.pkl')
    else:
        classes = pickle.load(Path(args.classes_pickle).open('rb'))
        valid_data = pickle.load(Path(args.valid_split_pickle).open('rb'))
    labelled_data = pickle.load(Path(args.train_split_pickle).open('rb'))
    # dataset_path = os.path.join(args.dataset_path, f"external/{dataset_name}/{dataset_name}_full/")
    dataset_path = os.path.join(args.dataset_path, dataset_name)
    dataset_meta = {'classes': classes}
    if args.no_imgnet_pretrained or (args.model_backbone is not None and 'resnet' not in args.model_backbone):
        dataset_meta['stats'] = STATS[dataset_name]
        print(f'Using {dataset_name} stats for normalization')
    else:
        dataset_meta['stats'] = STATS['imagenet']
        print(f'Using imagenet stats for normalization')


    setattr(args, 'dataset_path', dataset_path)
    L = len(labelled_data)

    model = SemCo(args, dataset_meta, device, L)
    model.train(labelled_data=labelled_data, valid_data=valid_data, save_best_model=True)