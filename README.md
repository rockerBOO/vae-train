# VAE Train

<!--toc:start-->
- [VAE Train](#vae-train)
  - [Install](#install)
  - [Usage](#usage)
    - [Advanced configuration](#advanced-configuration)
    - [GPU usage](#gpu-usage)
    - [Shell script](#shell-script)
  - [FAQ](#faq)
    - [NaN when running](#nan-when-running)
  - [Goals](#goals)
  - [Contribute](#contribute)
    - [Development](#development)
  - [TODO](#todo)
<!--toc:end-->

Trainer using diffusers AutoencoderKL for training VAEs.

- Compatible with HF datasets
- Accelerate for performance and distributed training
- Metrics using Tensorboard, Wandb
- Compatible with HF diffusers models like Stable Diffusion, Flux

## Install

Install [uv](https://docs.astral.sh/uv/) for package management.

Using uv, sync the dependencies.

```
uv sync --frozen --no-install-project --no-dev
```

extras: xformers, wandb, bitsandbytes. See pyproject.toml for complete list.

_NOTE_ if you are having any issues, let me know. I am trying out `uv` but interested in any issues with syncing dependencies or usage.

## Usage

```
accelerate launch train.py --pretrained_model_name_or_path stabilityai/sd-vae-ft-mse --dataset_name v-xchen-v/celebamask_hq
```

### Advanced configuration

```bash
accelerate launch train.py --pretrained_model_name_or_path stabilityai/sd-vae-ft-mse --dataset_name v-xchen-v/celebamask_hq --gradient_checkpointing  --gradient_accumulation_steps=2 --report_to=tensorboard --train_only_decoder --checkpointing_steps=1000 --train_batch_size=1 --use_8bit_adam --mixed_precision no
```

### GPU usage

- `--train_only_decoder` - Ideal training for fine-tuning VAE used in generative diffusion.
- `--gradient_checkpointing` - roughly 60% less VRAM for about 25% slower training. Good compromise to fit on the GPU.
- `--use_8bit_adam` - Quantize the optimizer, saves some VRAM (haven't measured)
- `--mixed_precision` - Keeps everything at full precision, but will use mixed precision when doing some operations to save VRAM and increase operation performance. Good speed and VRAM performance.

### Shell script

Sample shell script showing arguments I pass for a training run.

```bash
model=stabilityai/sd-vae-ft-mse

uv run accelerate launch train.py --pretrained_model_name_or_path $model --dataset_name v-xchen-v/celebamask_hq --gradient_checkpointing --gradient_accumulation_steps=2 --report_to=tensorboard --train_only_decoder --checkpointing_steps=1000 --train_batch_size=1 --fuse_qkv_projections --xformers --learning_rate 1.5e-7 --lr_scheduler cosine
```

_NOTE_ Allowing you to enter all these as a single configuration file is a desired goal.

## FAQ

### NaN when running

If you are training the SDXL VAE or other models, it might have issues with smaller precision so make sure it's using fp32/float/full precision.

## Goals

Motivation is to create a trainer for training VAE models that is easy-to-use and extendable for future autoregressive training. Focusing on user experience through improved APIs and interfaces.

## Contribute

Looking for contributions to improve performance, more algorithms, interfaces, distribution through libraries, web interfaces.

### Development

- ruff (lint, format)
- type-hinted

## TODO

- allow VRAM metrics usage printing to be toggleable
- add evaluation/test performance metrics
- add inference tool
- add A/B diff evaluation
- add LoRA/PEFT properly train, inference, and merging
