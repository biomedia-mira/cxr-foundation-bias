# Risk of bias in chest radiography deep-learning foundation models

This repository contains the code for the paper
> Ben Glocker, Charles Jones, Mélanie Roschewitz, Stefan Winzeck  
> **Risk of bias in chest radiography deep-learning foundation models**
> 2023. Under review

## Dataset

The CheXpert imaging dataset can be downloaded from https://stanfordmlgroup.github.io/competitions/chexpert/. The corresponding demographic information is available [here](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf).

## CXR Foundation Model

In our work we analyze the CXR foundation model by Google Health. Information on how to use the model to generate feature embeddings for the CheXpert dataset is available on the [original GitHub repository](https://github.com/Google-Health/imaging-research/tree/master/cxr-foundation).

## Code

For running the code, we recommend setting up a dedicated Python environment.

### Setup Python environment using conda

Create and activate a Python 3 conda environment:

   ```shell
   conda create -n chexploration python=3
   conda activate chexploration
   ```
   
Install PyTorch using conda (for CUDA Toolkit 11.3):
   
   ```shell
   conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
   ```
   
### Setup Python environment using virtualenv

Create and activate a Python 3 virtual environment:

   ```shell
   virtualenv -p python3 <path_to_envs>/chexploration
   source <path_to_envs>/chexploration/bin/activate
   ```
   
Install PyTorch using pip:
   
   ```shell
   pip install torch torchvision
   ```
   
### Install additional Python packages:
   
   ```shell
   pip install matplotlib jupyter pandas seaborn pytorch-lightning scikit-learn scikit-image tensorboard tqdm openpyxl tabulate statsmodels
   ```

### Requirements

The code has been tested on Windows 10 and Ubuntu 18.04/20.04 operating systems. The data analysis does not require any specific hardware and can be run on standard laptop computers. The training and testing of the disease detection models requires a high-end GPU workstation. For our experiments, we used a NVIDIA Titan X RTX 24 GB.

### How to use

In order to replicate the results presented in the paper, please follow these steps:

1. Download the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/), copy the file `train.csv` to the `datafiles/chexpert` folder. Download the [CheXpert demographics data](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf), copy the file `CHEXPERT DEMO.xlsx` to the `datafiles/chexpert` folder.
2. Run the notebook [`chexpert.sample.ipynb`](notebooks/chexpert.sample.ipynb) to generate the study data.
3. Run the notebook [`chexpert.resample.ipynb`](notebooks/chexpert.resample.ipynb) to perform test-set resampling.

To train and analyze the CheXpert model:

1. Adjust the variable `img_data_dir` to point to the CheXpert imaging data and run the following scripts:
   - Run the script [`disease-prediction.chexpert-model.py`](prediction/disease-prediction.chexpert-model.py) to train the disease detection model.
2. Run the script [`evaluate_disease_detection.py`](notebooks/evaluate_disease_detection.py) to evaluate the prediction performance.
3. Run the notebook [`chexpert.bias-inspection.chexpert-model.ipynb`](notebooks/chexpert.bias-inspection.chexpert-model.ipynb) for the statistical bias analysis.

To train and analyze the CXR foundation model:

1. Adjust the variable `data_dir` to point to the CheXpert embeddings from the CXR foundation model and run the following scripts:
   - Run the script [`disease-prediction.cxr-foundation.py`](prediction/disease-prediction.cxr-foundation.py) to train the disease detection model. Default is set to a linear prediction head. Check the code for CXR-MLP-3 and CXR-MLP-5 variants.
2. Run the script [`evaluate_disease_detection.py`](notebooks/evaluate_disease_detection.py) to evaluate the prediction performance.
3. Run the notebook [`chexpert.bias-inspection.cxr-foundation.ipynb`](notebooks/chexpert.bias-inspection.cxr-foundation.ipynb) for the statistical bias analysis.

Note, for CXR foundation model it is assumed that all embeddings for the CheXpert dataset are already available and converted from TensforFlow to NumPy. See the notebook [`chexpert.convert.cxr-foundation.ipynb`](notebooks/chexpert.convert.cxr-foundation.ipynb) for how to convert the embeddings.

## Funding sources
This work is supported through funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant Agreement No. 757173, [Project MIRA](https://www.project-mira.eu), ERC-2017-STG) and by the [UKRI London Medical Imaging & Artificial Intelligence Centre for Value Based Healthcare](https://www.aicentre.co.uk/).

## License
This project is licensed under the [Apache License 2.0](LICENSE).
