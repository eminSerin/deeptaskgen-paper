# deeptaskgen-paper
This repository contains the code for the paper "Generating Synthetic Task-based Brain Fingerprints for Population Neuroscience using Deep Learning". All code is provided to reproduce the results presented in the paper from scratch.


Note: This repository includes the specific DeepTaskGen version used in the paper. The latest updates and versions can be found at https://github.com/eminSerin/DeepTaskGen.

## Installation
To download the code, use the following command to clone the repository:

```
git clone https://github.com/eminSerin/deeptaskgen-paper.git
```

We recommend using conda to manage package versions and create a virtual environment. First, install conda by following the instructions here: https://docs.conda.io/en/latest/miniconda.html. Then, create a new environment and install the required packages:

```
conda env create -n task-generation --python=3.10
pip install -r deeptaskgen-paper/requirements.txt
```

## Steps to Reproduce the Results

### 1. Download Datasets
The following datasets are used in this project: 
1. Human Connectome Project Young Adult (HCP-YA): Download the minimally preprocessed data from https://db.humanconnectome.org/data/projects/HCP_1200, or use the provided script `experiments/training/__hcp_datafetch.py` to download directly from their AWS bucket.
2. Human Connectome Project Development (HCP-D): The preprocessed data is available at https://www.humanconnectome.org/study/hcp-lifespan-development/data-releases.
3. UK Biobank (UKB): Access the public dataset at https://www.ukbiobank.ac.uk/.

### 2. Preparing Data for Training
You can find the data preparation scripts for each dataset within their respective directories (e.g., `experiments/training/preprocessing` or `experiments/transfer_learning/hcp_development/preprocessing`). While preprocessing varies slightly between datasets, the general steps are as follows:

1. Extract time-series data for each ICA-based ROI (50 in total) from resting-state fMRI data and compute Voxel-to-ROI connectome matrices for subjects.
2. Save task contrast maps (HCP 47 Contrast Maps: `experiments/utils/hcp-ya_contrasts.txt`) to a single .npy file. For HCP datasets, surface-based contrast maps (cope files) will be projected into MNI152 volume space using the Registration Fusion Method by [Wu et al., 2018](https://doi.org/10.1002/hbm.24213).
3. Perform additional processing steps for transfer learning to prepare the trained model for fine-tuning.

Corresponding directories for each input type (e.g., raw, rest, task) have been created as placeholders for raw and processed data.

### 3. Training the Main DeepTaskGen Model on HCP-YA
Once the data is prepared, you can train the main DeepTaskGen model on the HCP-YA dataset. The relevant scripts for each step can be found in the `experiments/training` directory. These scripts cover training DeepTaskGen, predicting contrast maps for unseen subjects, and comparing against baseline methods (linear model, group average maps, and retest scans). To visualize the results, use `experiments/training/5_plot_results.ipynb`.

### 4. Transfer Learning on HCP-D and UKB
After training and testing DeepTaskGen on HCP-YA, we assessed its generalizability by transferring the learned parameters to unseen datasets: HCP-D and UKB. The scripts for this section are numbered and located in the `experiments/transfer_learning` directory. Specifically, for HCP-D, we fine-tuned the model using either EMOTION FACES-SHAPES or GAMBLING REWARD contrast maps and then predicted the other contrast maps. To visualize the HCP-D results, use `experiments/transfer_learning/hcp_development/7_plot_results.ipynb`. For UKB, we only fine-tuned the model using EMOTION FACES-SHAPES as it is the only available fMRI task. 

### 5. Validation
The practical applicability of the predicted task contrast maps was assessed by predicting individuals' demographic, cognitive and clincial characteristics, including age, sex, fluid intelligence, dominant hand grip strength, overall health, depression status, neuroticism, alcohol use frequency, GAD-7, PHQ-9, and RDS-4. You can find the validation scripts in the `experiments/validation/3_prediction/` directory. Before validation, you can also visualize the average structure of the predicted contrast maps and explore between-subject and task variability using the scripts in the `experiments/validation/1_group_maps/` and `experiments/validation/2_umaps` directories, respectively.