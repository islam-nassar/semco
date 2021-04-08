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
We use a fixed format for datasets to enable running the code on any dataset of choice without the need to edit the dataloaders. All the datasets we use follow the below folder structure (illustrated for cifar100 datasets:
.
+-- cifar100
|   +-- test
|   +-- train
|   +-- labels

```
    $ mkdir -p dataset && cd data
    $ wget -c http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    $ tar -xzvf cifar-10-python.tar.gz
```

download cifar-100 dataset: 
```
    $ mkdir -p dataset && cd data
    $ wget -c http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    $ tar -xzvf cifar-100-python.tar.gz
```

## Train the model

To train the model on CIFAR10 with 40 labeled samples, you can run the script: 
```
    $ CUDA_VISIBLE_DEVICES='0' python train.py --dataset CIFAR10 --n-labeled 40 
```
To train the model on CIFAR100 with 400 labeled samples, you can run the script: 
```
    $ CUDA_VISIBLE_DEVICES='0' python train.py --dataset CIFAR100 --n-labeled 400 
```
