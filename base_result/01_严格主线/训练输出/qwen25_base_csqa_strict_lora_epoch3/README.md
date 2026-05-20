---
library_name: peft
license: other
base_model: /root/models/Qwen2.5-0.5B
tags:
- base_model:adapter:/root/models/Qwen2.5-0.5B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen25_base_csqa_strict_lora_epoch3
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen25_base_csqa_strict_lora_epoch3

This model is a fine-tuned version of [/root/models/Qwen2.5-0.5B](https://huggingface.co//root/models/Qwen2.5-0.5B) on the csqa_train_strict dataset.
It achieves the following results on the evaluation set:
- Loss: 3.1592

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 3.0418        | 0.4106 | 500  | 3.0052          |
| 3.0397        | 0.8213 | 1000 | 2.9757          |
| 2.8294        | 1.2316 | 1500 | 3.0144          |
| 2.8519        | 1.6422 | 2000 | 2.9917          |
| 2.7208        | 2.0526 | 2500 | 3.0864          |
| 2.8053        | 2.4632 | 3000 | 3.1689          |
| 2.7371        | 2.8738 | 3500 | 3.1598          |
| 2.7974        | 3.0    | 3654 | 3.1592          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.7.1+cu118
- Datasets 4.0.0
- Tokenizers 0.22.2