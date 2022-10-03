# BenchmarkPDA

BenchmarkPDA is a Pytorch framework to benchmark Partial Domain Adaptation methods with different model selection strategies.

## Prerequisites

* pytorch==1.8.0
* torchvision==0.9.
* torchaudio==0.8.0
* cudatoolkit=11.1
* pyyaml
* scikit-learn
* jupyterlab
* prettytable
* ipywidgets
* tqdm
* pandas
* opencv
* pot
* cvxpy


## Datasets

The currently supports the following datasets:
* Office-Home ([Venkateswara et al., 2017](https://arxiv.org/abs/1706.07522))
* VisDA ([Peng et al., 2017](https://arxiv.org/abs/1710.06924))

Place them in the folder `datasets`. Make sure to place the image path lists in the `image_list` in the respective dataset folder.


## Methods

The currently available methods are:
* Source Only
* PADA ([Cao et al., 2018a](https://arxiv.org/abs/1808.04205))
* SAFN ([Xu et al., 2019](https://arxiv.org/abs/1811.07456))
* BA3US ([Jian et al., 2020](https://arxiv.org/abs/2003.02541))
* AR ([Gu et al., 2021](https://proceedings.neurips.cc/paper/2021/file/7ce3284b743aefde80ffd9aec500e085-Paper.pdf))
* JUMBOT ([Fatras et al., 2021](https://arxiv.org/abs/2103.03606))
* MPOT ([Nguyen et al., 2022](https://arxiv.org/abs/2108.09645)).


## Step 1 - Hyper-Parameter Grid Search

To run the hyper-parameter search for the Office-Home dataset use

```bash
python hp_search_train_val.py --method METHOD --dset office-home --source_domain Art --target_domain Clipart
```

where METHOD should be one of the following: `source_only_plus`, `pada`, `ba3us`, `ar`, `jumbot`, `mpot`.

To run the hyper-parameter search for the VisDA dataset use

```bash
python hp_search_train_val.py --method METHOD --dset visda --source_domain train --target_domain validation
```

where METHOD should be one of the following: `source_only_plus`, `pada`, `ba3us`, `ar`, `jumbot`, `mpot`. The train domain are the synthetic images, while the validation domain corresponds to the real images.

## Step 2 - Select Best Hyper-Parameters

We select the best hyper-parameters using the Jupyter Notebook `Step 2 - Select Hyper-Parameter Grid Search.ipynb`. It includes code to recreate Tables 4 and 10.

## Step 3 - Train Best Hyper-Paramters

To train the models with the different hyper-parameters chosen use

```bash
python train_hp_chosen.py --dset DATASET
```

where DATASET should be `office-home` or `visda`.

## Step 4 - Collect results and generate tables

We gather all the results using the Jupyter Notebook `Step 4 - Collect Results.ipynb`. It includes code to generate Tables 1, 5, 6, 7 and 12.


## License

This source code is released under the MIT license, included [here](LICENSE).