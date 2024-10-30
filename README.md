# VAE Train

Trainer using diffusers AutoencoderKL for training VAEs.

- Compatible with HF datasets
- Accelerate for performance and distributed training
- Metrics using Tensorboard, Wandb
- Compatible with HF diffusers models like Stable Diffusion, Flux

## Usage

```
accelerate launch train.py --pretrained_model_name_or_path stabilityai/sdxl-vae
```

### Advanced configuration

```bash
accelerate launch train.py --pretrained_model_name_or_path stabilityai/sdxl-vae --gradient_checkpointing  --gradient_accumulation_steps=2 --report_to=tensorboard --train_only_decoder --checkpointing_steps=1000 --train_batch_size=1 --use_8bit_adam --mixed_precision no
```

### GPU usage

- `--train_only_decoder` - Ideal training for fine-tuning VAE used in generative diffusion.
- `--gradient_checkpointing` - roughly 60% less VRAM for about 25% slower training. Good compromise to fit on the GPU.
- `--use_8bit_adam` - Quantize the optimizer, saves some VRAM (haven't measured)
- `--mixed_precision` - Keeps everything at full precision, but will use mixed precision when doing some operations to save VRAM and increase operation performance. Good speed and VRAM performance.
