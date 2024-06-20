---
license: apache-2.0
library_name: peft
tags:
- generated_from_trainer
base_model: distilbert-base-uncased
metrics:
- accuracy
model-index:
- name: distilbert-base-uncased-lora-text-classification
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert-base-uncased-lora-text-classification

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8136
- Accuracy: {'accuracy': 0.893}

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy            |
|:-------------:|:-----:|:----:|:---------------:|:-------------------:|
| No log        | 1.0   | 125  | 0.2775          | {'accuracy': 0.887} |
| No log        | 2.0   | 250  | 0.4787          | {'accuracy': 0.864} |
| No log        | 3.0   | 375  | 0.4367          | {'accuracy': 0.887} |
| 0.2436        | 4.0   | 500  | 0.5377          | {'accuracy': 0.889} |
| 0.2436        | 5.0   | 625  | 0.8811          | {'accuracy': 0.874} |
| 0.2436        | 6.0   | 750  | 0.7428          | {'accuracy': 0.891} |
| 0.2436        | 7.0   | 875  | 0.7456          | {'accuracy': 0.883} |
| 0.0283        | 8.0   | 1000 | 0.8222          | {'accuracy': 0.893} |
| 0.0283        | 9.0   | 1125 | 0.8001          | {'accuracy': 0.892} |
| 0.0283        | 10.0  | 1250 | 0.8136          | {'accuracy': 0.893} |


### Framework versions

- PEFT 0.11.1
- Transformers 4.41.2
- Pytorch 2.3.1+cu121
- Datasets 2.19.2
- Tokenizers 0.19.1