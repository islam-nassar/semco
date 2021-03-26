import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import os


def generate_pickles(ds_name, data_labels_path, output_path, instances_per_label, generate_cls_valid, seed):
    path = Path(data_labels_path)
    train_labels = pd.read_feather(path / 'labels_train.feather')
    test_labels = pd.read_feather(path / 'labels_test.feather')
    test_labels.id = 'test/' + test_labels.id
    train_labels.id = 'train/' + train_labels.id
    classes = sorted(list(set(train_labels['class'].values)))
    if generate_cls_valid:
        valid_dict = {k: v for k, v in zip(test_labels.id.values, test_labels['class'].values)}
        pickle.dump(valid_dict, Path(output_path + f'{ds_name}_valid_data.pkl').open('wb'))
        pickle.dump(classes, Path(output_path + f'{ds_name}_classes.pkl').open('wb'))
        print('Generated classes and test/valid pickles')

    np.random.seed(seed)
    labelled_data = {}
    for cls in classes:
        filenames = train_labels[train_labels['class'] == cls].id.values
        if len(filenames) <= instances_per_label:
            print(f'{cls} class only has {len(filenames)} instances')
            choices = filenames
        else:
            choices = np.random.choice(filenames, size=instances_per_label, replace=False)
        lbs = {elem: cls for elem in choices}
        labelled_data.update(lbs)

    filepath = output_path + f'{ds_name}_labelled_data_{instances_per_label}_seed{seed}.pkl'
    pickle.dump(labelled_data, Path(filepath).open('wb'))
    print(f'Generated labelled data pickle: {filepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Test pickle generation')
    parser.add_argument('--dataset-path', type=str, default=os.environ.get('SEMCO_DATA_PATH', '/home/inas0003/data'),
                        help='the path to the data folder containing datasets')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='dataset name (ex: cifar100)')
    parser.add_argument('--instances-per-label', type=int, default=None,
                        help='instances per label for the splits')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='random seed for the split')
    parser.add_argument('--generate-classes-and-valid', type=bool, default=False,
                        help='if set to true, classes and test pickels will be generated')
    args, _ = parser.parse_known_args()

    ds_name = args.dataset_name
    data_labels_path = os.path.join(args.dataset_path, f'{ds_name}/labels/')
    output_path = '../splits/'
    instances_per_label = args.instances_per_label

    generate_pickles(ds_name, data_labels_path, output_path, instances_per_label,
                     generate_cls_valid=args.generate_classes_and_valid, seed=args.random_seed)