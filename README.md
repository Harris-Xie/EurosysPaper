# Root Cause Analysis (RCA) for Microsoft Services

This repository contains scripts and data for conducting Root Cause Analysis (RCA) on Microsoft Services. The following is a description of the directory structure and the purpose of each script.

## Directory Structure

- `rca/`
  - `Data_Models/`: Contains the raw data and preprocessed data used in the analysis.
    - `utf8allcharnobom.jsonl`: Raw data file.
  - `evaluation.py`: Script used for parsing and evaluating the output of the inference.
  - `Fasttext.py`: Baseline script for training and using the FastText model.
  - `FasttextTrain.py`: Script used for training the FastText model.
  - `predict.py`: Script used for inferring using the trained model and GPT prompt.
  - `summary.py`: Script used for preprocessing the raw data and summarizing the AdviceDetail.
  - `TrainTestSplit.py`: Script used for splitting the preprocessed data into train and test sets.
  - `utils.py`: Commonly used methods stored here.
  - `XGboost.py`: Baseline script for training and using the XGBoost model.

## Preprocessing

### summary.py

- This script is used for preprocessing the raw data and summarizing the AdviceDetail.
- Input: `Data_Models/utf8allcharnobom.jsonl`
- Output: `Data_Models/all_info_gpt4_sum.jsonl`

### TrainTestSplit.py

- This script is used for splitting the preprocessed data into train and test sets.
- Input: `Data_Models/all_info_gpt4_sum.jsonl`
- Output: `Data_Models/test.jsonl`, `Data_Models/train.jsonl`

### FasttextTrain.py

- This script is used for training the FastText model.
- Input: `Data_Models/train.jsonl`
- Output: `Data_Models/train_models`

## Inferring

### predict.py

- This script is used for inferring using the trained model and GPT prompt.
- Input: `Data_Models/...` (specific input files)
- Output: `Data_Models/y_pred.json`, `Data_Models/y_true.json`, `Data_Models/y_generator.json`, ...

## Parsing and Evaluation

### evaluation.py

- This script is used for parsing and evaluating the output of the inference.
- Input: `Data_Models/output_{}_{}` (placeholder for specific input files)
- Output: Evaluation metrics

Please note that the provided information is a high-level overview of the different scripts and their inputs/outputs.



