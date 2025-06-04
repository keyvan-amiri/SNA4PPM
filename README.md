# QueueTab: Remaining Time Prediction via Queuing Networks and Machine Learning for Tabular Data

This is the github repository for our paper "QueueTab: Remaining Time Prediction via Queuing Networks and Machine Learning for Tabular Data" submitted to ICPM2025 conference.

<p align="center">
  <img src="https://github.com/keyvan-amiri/SNA4PPM/blob/main/QueueTab.jpg" width="600">
</p>


## Installation

### Feature Extraction
The feature extraction process involves three main steps:

- Establishing the activity-instance log

- Creating the queuing network

- Extracting intra- and inter-case features

To execute these steps, specify the dataset name and run the **Prepare_log.py** script as shown below:
```bash
python Prepare_log.py --dataset BPIC20DD
```

#### Training Tabular models

##### Training baseline models

Name of the target column is "**rem_time**" for all datasets.

Columns that should not be used for predictions are the followings:
"**case:concept:name**", "**prefix_length**", "**set**": these are always the first three columns, the first two show the case id and its length and used to specify for which instance the prediction is done. At last we need a dataframe for predictions with minimum 4 columns: "case:concept:name", "prefix_length" the real observed remaining time ("rem_time") and the prediction of the model. The column set determines whether this row belongs to train or test set. (no validation set is specified so feel free to create a one)

columns "**start**", "**end**", "**enabled_time**" are raw timestamps that are used for feature extraction.

"**next_proc**", "**next_wait**" are two additional target columns for next processing and next waiting time prediction. But, after rethinking the story of the paper, I think it is better to only focus on remaining time prediction. Therefore, these two columns should not be used for predictions or as target columns.

