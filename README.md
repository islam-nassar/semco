# SemCo
The official implementation of the paper All Labels Are Not Created Equal: Enhancing Semi-supervision via Label Grouping and Co-training

## Install Dependencies

- Create a new environment and install dependencies using ```pip install -r requirements.txt```
- Install apex to enable automatic mixed precision training (AMP).
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext

```
**Note**: Installing apex is optional, if you don't want to implement amp, you can simply pass `--no_amp` command line argument to the launcher. 


## Dataset
We use a standard directory structure for all our datasets to enable running the code on any dataset of choice without the need to edit the dataloaders. The datasets directory follow the below structure (only shown for cifar100 but is the same for all other datasets):
```
datasets
└───cifar100
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
```
An example of the above directory structure for cifar100 can be found here. 

To preprocess a generic dataset into the above format, you can refer to `create_dataset_structure.py` for several examples. 

To configure the datasets directory path, you can either set the environment variable `SEMCO_DATA_PATH` or pass a command line argument `--dataset-path` to the launcher. (e.g. `export SEMCO_DATA_PATH=/home/data`). Note that this path references the parent datasets directory which contains the different sub directories for the individual datasets (e.g. cifar100, mini-imagenet, etc.)

## Label Semantics Embeddings
SemCo expects a prior representation of all class labels via a semantic embedding for each class name. In our experiments, we use embeddings obtained from ConceptNet knowledge graph which contains a total of ~550K term embeddings. SemCo uses a matching criteria to find the best embedding for each of the class labels. Alternatively, you can use class attributes as the prior (like we did for CUB200 dataset), so you can build your own semantic dictionary.

To run experiments, please download the semantic embedding file here and set the path to the downloaded file either via `SEMCO_WV_PATH` environment variable or `--word-vec-path` command line argument. (e.g. `export SEMCO_WV_PATH=/home/inas0003/data/numberbatch-en-19.08_128D.dict.pkl`

## Defining the Splits
For each of the experiments, you will need to specify to the launcher 4 command line arguments:
- `--dataset-name`: denoting the dataset directory name (e.g. cifar100)
- `--train-split-pickle`: path to pickle file with training split
- `--valid-split-pickle`: (optional) path to pickle file with validation/test split (by default contains all the files in the `test` folder) 
- `--classes-pickle`: (optional) path to pickle file with list of class names

To obtain the three pickle files for any dataset, you can use `generate_tst_pkls.py` script specifying the dataset name and the number of instances per label and optionally a random seed. Example as follows:

`python generate_tst_pkls.py --dataset-name cifar100 --instances-per-label 10 --random-seed 000 --output-path splits`

The above will generate a train split with 10 images per class using a random seed of 000 together with the class names and the validation split containing all the files placed in the `test` folder. This can be tweaked by editing the python script. 

## Training the model

To train the model on cifar100 with 40 labeled samples, you can run the script: 
```
    $ python launch_semco.py --dataset-name cifar100 --train-split-pickle splits/cifar100_labelled_data_40_seed123.pkl --model_backbone=wres --wres-k=2
```
or without amp
```
    $ python launch_semco.py --dataset-name cifar100 --train-split-pickle splits/cifar100_labelled_data_40_seed123.pkl --model_backbone=wres --wres-k=2 --no_amp
```
Similary to train the model on mini_imagenet with 400 labeled samples, you can run the script: 
```
    $  python launch_semco.py --dataset-name mini_imagenet --train-split-pickle testing/mini_imagenet_labelled_data_40_seed456.pkl --model_backbone=resnet18 --im-size=84 --cropsize=84 
```
