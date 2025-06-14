# QueueTab: Remaining Time Prediction via Queuing Networks and Machine Learning for Tabular Data

This is the repository for our paper "QueueTab: Remaining Time Prediction via Queuing Networks and Machine Learning for Tabular Data" submitted to ICPM2025 conference.

<p align="center">
  <img src="https://github.com/keyvan-amiri/SNA4PPM/blob/main/QueueTab.jpg" width="600">
</p>


## Installation
First, clone this GitHub repository to your local machine:

```bash
git clone https://github.com/keyvan-amiri/SNA4PPM
```
To install and set up the required environment on a Linux system, run the following commands:

```bash
conda create -n queue_tab python=3.11
conda activate queue_tab
pip install -r requirements.txt
conda clean --all
```
## Feature Extraction
The feature extraction process involves three main steps:

- Establishing the activity-instance log

- Creating the queuing network

- Extracting intra- and inter-case features

To execute these steps, specify the dataset name and run the **Prepare_log.py** script as shown below:
```bash
python Prepare_log.py --dataset BPIC20DD
```
This process generates a **.csv** file used to train tabular models such as TabM and CatBoost. For example, the output of the previous script is [BPIC20DD_two_TS.csv](https://github.com/keyvan-amiri/SNA4PPM/blob/main/data/processed/BPIC20DD/BPIC20DD_two_TS.csv). All processed datasets are already available in the repository, so you may skip this step and proceed directly to training the tabular models. The structure of the **.csv** file is consistent across all datasets. The first three columns are:

- **case:concept:name** indicates case ID

- **prefix_length**  indicates prefix length of the example

- **set** indicates whether the example belongs to the training, validation, or test set

The target column for remaining time prediction is **rem_time**. Additional timestamp columns (start, end, enabled_time) are used during feature extraction but not for training the tabular models. Two extra columns, next_proc and next_wait, are included for predicting next processing and waiting times, though these tasks are beyond the scope of our paper.

To run experiments on other event logs, create a separate configuration file similar to [BPIC20DD.yaml](https://github.com/keyvan-amiri/SNA4PPM/blob/main/cfg/BPIC20DD.yaml).

## Training Tabular Models
After feature extraction, tabular models are trained using the **run_tabular_experiment.py** script. You can replicate our experiments across all 10 event logs in the paper using the **run_tabular_experiment.sh** bash script. A summary of the results for each dataset is available [here](https://github.com/keyvan-amiri/SNA4PPM/tree/main/results_compact). For evaluation and additional visualizations, refer to the **Evaluation.ipynb** Jupyter notebook. 

## Training Baseline Models
To train and evaluate baseline approaches **LS-ICE** and **PGTNet** follow the instruction [here](https://github.com/keyvan-amiri/SNA4PPM/blob/main/baselines/README.md). 
