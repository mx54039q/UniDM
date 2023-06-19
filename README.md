# UniDM: A Unified Framework for Data Manipulation with Large Language Models

This is the official repo for *UniDM: A Unified Framework for Data Manipulation with Large Language Models*.

## Install

#### Download and unpack the dataset:
```
mkdir dataset
wget https://fm-data-tasks.s3.us-west-1.amazonaws.com/datasets.tar.gz -P dataset
tar xvf dataset/datasets.tar.gz -C dataset/
```

#### Prerequisites

```
pip install -r requirements.txt

# Manifest
git clone git@github.com:HazyResearch/manifest.git
cd manifest
pip install -e .

```

## Run
To run inference, use
```
python inference.py --help
```
Some examples are as follows:
(The API KEY can be obtained by registering with the LLM provider. For instance, if you want to run inference with the OpenAI API models, create an account [here](https://openai.com/api/).)

To run data imputation task select 3 example to add to the prompt by the auto-retrieve module, using both the adaptive data parsing and prompt engineering modules,
```
python inference.py \
    --api_key <YOUR API KEY> \
    --data_dir <DATA DIR> \
    --task data_imputation \
    --instance_num 3 \
    --metadata_wise \
    --instance_wise \
    --data_parsing \
    --prompt_engineering
```
To run entity resolution task and select 3 examples to add to the prompt, using both the adaptive data parsing and prompt engineering modules,
```
python inference.py \
    --api_key <YOUR API KEY> \
    --data_dir <DATA DIR> \
    --task entity_resolution \
    --context_num 3 \
    --metadata_wise \
    --instance_wise \
    --data_parsing \
    --prompt_engineering
```
To run data transformation task and select 3 examples to add to the prompt, using both the adaptive data parsing and prompt engineering modules,
```
python inference.py \
    --api_key <YOUR API KEY> \
    --data_dir <DATA DIR> \
    --task data_transformation \
    --context_num 3 \
    --data_parsing \
    --prompt_engineering
```

## Notes

The data retrieval may take time. When inference, we restore the retrieval scores in the `ret_score` folder intermediately. And we provide the retrieval scores of examples above for quick verification.







