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
- name: qwen25_base_csqa_tmpl_lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen25_base_csqa_tmpl_lora

This model is a fine-tuned version of [/root/models/Qwen2.5-0.5B](https://huggingface.co//root/models/Qwen2.5-0.5B) on the csqa_train_tmpl dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1182

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
- num_epochs: 2.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.1325        | 0.4106 | 500  | 1.1218          |
| 1.1295        | 0.8213 | 1000 | 1.1127          |
| 1.0593        | 1.2316 | 1500 | 1.1197          |
| 1.0658        | 1.6422 | 2000 | 1.1190          |
| 1.0722        | 2.0    | 2436 | 1.1182          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.7.1+cu118
- Datasets 4.0.0
- Tokenizers 0.22.2