#
# Copyright 2024 Dave Lage
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os

from diffusers.models.modeling_utils import ModelMixin
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available, is_wandb_available
from packaging import version
from peft.mapping import get_peft_model
from peft.tuners.lora.config import LoraConfig
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torchvision import transforms
from tqdm import tqdm

from vae_train.trainer_args import parse_args, TrainerArgs
from vae_train.trace import TorchTracemalloc, b2mb


logger = get_logger(__name__, log_level="INFO")


def prepare_dataset(args: TrainerArgs, accelerator):
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name != "":
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )

    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    assert isinstance(dataset, IterableDatasetDict) or isinstance(dataset, DatasetDict)

    train_dataset = dataset["train"]

    assert isinstance(train_dataset, Dataset)

    column_names = train_dataset.column_names
    assert type(column_names) is list and len(column_names) > 0

    if args.image_column == "":
        image_column = column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )

    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples

    # def test_preprocess(examples):
    #     images = [image.convert("RGB") for image in examples[image_column]]
    #     examples["pixel_values"] = [test_transforms(image) for image in images]
    #     return examples

    with accelerator.main_process_first():
        # Load test data from test_data_dir
        if args.test_data_dir is not None and args.train_data_dir is not None: 
            logger.info(f"load test data from {args.test_data_dir}")
            test_dir = os.path.join(args.test_data_dir, "**")
            test_dataset = load_dataset(
                "imagefolder",
                data_files=test_dir,
                cache_dir=args.cache_dir,
            )

            assert isinstance(test_dataset, DatasetDict) or isinstance(
                test_dataset, IterableDatasetDict
            )

            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess)
            test_dataset = test_dataset["train"].with_transform(preprocess)
        # Load train/test data from train_data_dir
        elif isinstance(dataset, IterableDatasetDict) and "test" in dataset.keys():
            train_dataset = dataset["train"].with_transform(preprocess)
            test_dataset = dataset["test"].with_transform(preprocess)
        # Split into train/test
        else:
            dataset = dataset["train"].train_test_split(test_size=args.test_samples)
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess)
            test_dataset = dataset["test"].with_transform(preprocess)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}

    # Convert the dataset to torch specific for DataLoader
    # Then we ignore the type. Maybe the type could be better inferred.
    # train_dataset = train_dataset.with_format("torch")
    # test_dataset = test_dataset.with_format("torch")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,  # type: ignore[arg-type]
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.train_batch_size * accelerator.num_processes,
    )

    test_dataloader = DataLoader(
        test_dataset,  # type: ignore[arg-type]
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_workers
        or args.train_batch_size * accelerator.num_processes,
    )

    assert hasattr(train_dataset, "__len__")
    assert hasattr(test_dataset, "__len__")

    return train_dataloader, test_dataloader, len(train_dataset), len(test_dataset)  # type: ignore[arg-type]


# https://github.com/kukaiN/vae_finetune/blob/main/vae_finetune.py
# Function to split the image into patches
def extract_patches(image, patch_size, stride):
    # Unfold the image into patches
    patches = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    # Reshape to get a batch of patches
    patches = patches.contiguous().view(
        image.size(0), image.size(1), -1, patch_size, patch_size
    )
    return patches


# Patch-Based MSE Loss
def patch_based_mse_loss(real_images, recon_images, patch_size=32, stride=16):
    real_patches = extract_patches(real_images, patch_size, stride)
    recon_patches = extract_patches(recon_images, patch_size, stride)
    mse_loss = F.mse_loss(real_patches, recon_patches)
    return mse_loss


# Patch-Based LPIPS Loss (using the pre-defined LPIPS model)
def patch_based_lpips_loss(
    lpips_model, real_images, recon_images, patch_size=32, stride=16
):
    with torch.no_grad():
        real_patches = extract_patches(real_images, patch_size, stride)
        recon_patches = extract_patches(recon_images, patch_size, stride)

        lpips_loss = 0
        # Iterate over each patch and accumulate LPIPS loss
        for i in range(real_patches.size(2)):  # Loop over number of patches
            real_patch = real_patches[:, :, i, :, :].contiguous()
            recon_patch = recon_patches[:, :, i, :, :].contiguous()
            patch_lpips_loss = lpips_model(real_patch, recon_patch).mean()

            # Handle non-finite values
            if not torch.isfinite(patch_lpips_loss):
                patch_lpips_loss = torch.tensor(0, device=real_patch.device)

            lpips_loss += patch_lpips_loss

    return lpips_loss / real_patches.size(2)  # Normalize by the number of patches


@torch.no_grad()
def log_validation(args, repo_id, test_dataloader: DataLoader, vae, accelerator, epoch):
    logger.info("Running validation... ")

    vae_model = accelerator.unwrap_model(vae)

    assert isinstance(vae_model, AutoencoderKL)
    images = []

    for _, sample in enumerate(test_dataloader):
        x = sample["pixel_values"]

        assert isinstance(x, torch.Tensor)

        reconstructions = vae_model(x).sample

        assert isinstance(reconstructions, torch.Tensor)
        images.append(torch.cat([x.cpu(), reconstructions.cpu()]))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "Original (left), Reconstruction (right)", np_images, epoch
            )
        elif tracker.name == "wandb":
            import wandb

            tracker.log(
                {
                    "Original (left), Reconstruction (right)": [
                        wandb.Image(torchvision.utils.make_grid(image))
                        for _, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.gen_images}")

    if repo_id is not None and repo_id != "":
        save_model_card(args, repo_id, images, repo_folder=repo_id)


def make_image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def save_model_card(args, repo_id: str, images=None, repo_folder=None):
    img_str = ""
    if images is None or repo_folder is None:
        return

    if len(images) > 0:
        image_grid = make_image_grid(images, 1, "example")
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {args.pretrained_model_name_or_path}
datasets:
- {args.dataset_name}
tags:
- diffusers
- stable-diffusion-diffusers
- vae
inference: true
---
    """
    model_card = f"""
# VAE finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. \n

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    wandb_run_url = None

    if is_wandb_available():
        import wandb

        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def train_step(
    args: TrainerArgs,
    lpips_loss_fn: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    accelerator: Accelerator,
    vae: AutoencoderKL,
    mut_train_loss: float | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["pixel_values"]

    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py
    if accelerator.num_processes > 1:
        posterior = vae.module.encode(target).latent_dist
        z = posterior.sample()
        pred = vae.module.decode(z).sample
    else:
        encoding = vae.encode(target)
        posterior = encoding.latent_dist  # type: ignore[arg-type]
        # z = mean                      if posterior.mode()
        # z = mean + variable*epsilon   if posterior.sample()
        z = posterior.sample()  # Not mode()
        decoding = vae.decode(z)

        pred = decoding.sample  # type: ignore[arg-type]

    kl_loss = posterior.kl().mean()

    mse_loss = F.mse_loss(pred.float(), batch["pixel_values"].float(), reduction="mean")

    if args.patch_loss:
        # patched loss
        mse_loss = patch_based_mse_loss(
            target, pred, args.patch_size, args.patch_stride
        )
        lpips_loss = patch_based_lpips_loss(
            lpips_loss_fn,
            target,
            pred,
            args.patch_size,
            args.patch_stride,
        )

    else:
        # default loss
        mse_loss = F.mse_loss(pred, target, reduction="mean")
        with torch.no_grad():
            lpips_loss = lpips_loss_fn(pred, target).mean()
            if not torch.isfinite(lpips_loss):
                lpips_loss = torch.tensor(0)

    if args.train_only_decoder:
        # remove kl term from loss, bc when we only train the decoder, the latent is untouched
        # and the kl loss describes the distribution of the latent
        loss = mse_loss + args.lpips_scale * lpips_loss
    else:
        loss = mse_loss + args.lpips_scale * lpips_loss + args.kl_scale * kl_loss

    gathered = accelerator.gather(loss.repeat(args.train_batch_size))

    assert isinstance(gathered, torch.Tensor)

    avg_loss = gathered.mean()
    if mut_train_loss is not None:
        mut_train_loss += avg_loss.item() / args.gradient_accumulation_steps
    else:
        mut_train_loss = avg_loss.item() / args.gradient_accumulation_steps

    assert isinstance(loss, torch.Tensor)

    mse_loss = mse_loss.detach().item()
    lpips_loss = lpips_loss.detach().item()
    kl_loss = kl_loss.detach().item()

    assert type(mse_loss) is float
    assert type(lpips_loss) is float
    assert type(kl_loss) is float

    return loss, {
        "mse": mse_loss,
        "lpips": lpips_loss,
        "kl": kl_loss,
    }


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    accelerator: Accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    # Load vae
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        # subfolder="vae",
        revision=args.revision,
    )

    assert isinstance(vae, AutoencoderKL)

    ema_vae: EMAModel = EMAModel(
        vae.parameters(), model_cls=AutoencoderKL, model_config=vae.config
    )

    if args.fuse_qkv_projections:
        vae.fuse_qkv_projections()

    vae.requires_grad_(True)

    if args.lora:
        peft_config = LoraConfig(
            # check modules for Linear or Conv2D which can be used in LoRA
            target_modules=["conv1", "conv2", "to_q", "to_k", "to_v"],
            inference_mode=False,
            r=4,
            lora_alpha=16,
            lora_dropout=0.1,
            use_rslora=True,
        )

        assert isinstance(vae, ModelMixin)
        # We ensure that vae is valid to be used with get_peft_model. 
        # Requires PreTrainedModel and AutoencoderKL does not use it
        vae = get_peft_model(vae, peft_config) # type: ignore[arg-type]
        vae.print_trainable_parameters()

    if args.gradient_checkpointing:
        # args.use_lora or args.use_boft or args.use_oft:
        # text_encoder.gradient_checkpointing_enable()
        vae.enable_gradient_checkpointing()

    if args.xformers:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            vae.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.train_only_decoder:
        # freeze the encoder weights
        for param in vae.encoder.parameters():
            param.requires_grad_(False)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes` or `pip install bitsandbytes-windows` for Windows"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = AdamW

    vae.train()

    params_to_optimize = [param for param in vae.parameters() if param.requires_grad]

    if args.use_came:
        from came_pytorch import CAME

        optimizer_cls = CAME
        optimizer = optimizer_cls(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(0.9, 0.999, 0.9999),
            weight_decay=0.01,
        )
    else:
        optimizer = optimizer_cls(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    train_dataloader, test_dataloader, train_dataset_len, test_dataset_len = (
        prepare_dataset(args, accelerator)
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_train_epochs * args.gradient_accumulation_steps,
    )

    lpips_loss_fn = lpips.LPIPS(net="alex")
    lpips_loss_fn.requires_grad_(False)

    (
        vae,
        optimizer,
        train_dataloader,
        test_dataloader,
        lpips_loss_fn,
        lr_scheduler,
    ) = accelerator.prepare(
        vae,
        optimizer,
        train_dataloader,
        test_dataloader,
        lpips_loss_fn,
        lr_scheduler,
    )

    if args.lora:
        assert isinstance(vae.model, AutoencoderKL)
    else:
        assert isinstance(vae, AutoencoderKL)
    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(test_dataloader, DataLoader)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # ------------------------------ TRAIN ------------------------------ #
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {train_dataset_len}")
    logger.info(f"  Num test samples = {test_dataset_len}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    global_step, first_epoch, resume_step = maybe_resume_from_checkpoint(
        args, accelerator, num_update_steps_per_epoch
    )

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    train_loss: float | None = None

    with accelerator.autocast():
        for epoch in range(first_epoch, args.num_train_epochs):
            with TorchTracemalloc() as tracemalloc:
                for step, batch in enumerate(train_dataloader):
                    with accelerator.accumulate(vae):
                        loss, meta_loss = train_step(
                            args, lpips_loss_fn, batch, accelerator, vae, train_loss
                        )

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                filter(lambda m: m.requires_grad, vae.parameters()),
                                args.max_grad_norm,
                            )
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        if args.use_ema:
                            ema_vae.step(vae.parameters())
                        progress_bar.update(1)
                        global_step += 1
                        accelerator.log({"train_loss": train_loss}, step=global_step)
                        train_loss = 0.0

                        if global_step % args.checkpointing_steps == 0:
                            if accelerator.is_main_process:
                                save_path = os.path.join(
                                    args.output_dir, f"checkpoint-{global_step}"
                                )
                                accelerator.save_state(save_path)
                                logger.info(f"Saved state to {save_path}")

                    logs = {
                        "step_loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    } | meta_loss

                    accelerator.log(logs)
                    progress_bar.set_postfix(**logs)
                    break

                if accelerator.is_main_process:
                    if epoch % args.validation_epochs == 0:
                        with torch.no_grad():
                            log_validation(
                                args,
                                args.repo_id,
                                test_dataloader,
                                vae,
                                accelerator,
                                epoch,
                            )

            memory_usage_metrics(accelerator, tracemalloc)

        # Create the pipeline using the trained modules and save it.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            vae = accelerator.unwrap_model(vae)
            if args.use_ema:
                ema_vae.copy_to(vae.parameters())
            vae.save_pretrained(args.output_dir)

    accelerator.end_training()


def memory_usage_metrics(accelerator: Accelerator, tracemalloc: TorchTracemalloc):
    # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
    accelerator.print(
        f"GPU Memory before entering the train : {b2mb(tracemalloc.begin)}MB"
    )
    accelerator.print(
        f"GPU Memory consumed at the end of the train (end-begin): {tracemalloc.used}MB"
    )
    accelerator.print(
        f"GPU Peak Memory consumed during the train (max-begin): {tracemalloc.peaked}MB"
    )
    accelerator.print(
        f"GPU Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}MB"
    )

    accelerator.print(
        f"CPU Memory before entering the train : {b2mb(tracemalloc.cpu_begin)}MB"
    )
    accelerator.print(
        f"CPU Memory consumed at the end of the train (end-begin): {tracemalloc.cpu_used}MB"
    )
    accelerator.print(
        f"CPU Peak Memory consumed during the train (max-begin): {tracemalloc.cpu_peaked}MB"
    )
    accelerator.print(
        f"CPU Total Peak Memory consumed during the train (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}MB"
    )


def maybe_resume_from_checkpoint(args, accelerator, num_update_steps_per_epoch):
    global_step = 0
    first_epoch = 0
    resume_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            # dirs = os.listdir(args.output_dir)
            dirs = os.listdir(args.resume_from_checkpoint)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(path))  # kiml
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    return global_step, first_epoch, resume_step


if __name__ == "__main__":
    main()
