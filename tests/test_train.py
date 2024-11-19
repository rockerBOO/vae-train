import pytest
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from unittest.mock import Mock, patch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from train import (
    extract_patches,
    patch_based_mse_loss,
    patch_based_lpips_loss,
    train_step,
    prepare_dataset,
)
from vae_train.trainer_args import TrainerArgs


@pytest.fixture
def mock_args():
    args = Mock(spec=TrainerArgs)
    args.train_batch_size = 2
    args.resolution = 64
    args.patch_size = 32
    args.patch_stride = 16
    args.patch_loss = True
    args.lpips_scale = 0.5
    args.kl_scale = 1e-6
    args.train_only_decoder = False
    args.gradient_accumulation_steps = 1
    args.dataset_name = "v-xchen-v/celebamask_hq"
    args.dataset_config_name = ""
    args.cache_dir = "/tmp/x-s92-vae-train"
    args.train_data_dir = ""
    args.test_samples = 2
    args.image_column = "image"
    args.test_data_dir = None
    args.dataloader_workers = 0
    return args


@pytest.fixture
def mock_batch():
    return {"pixel_values": torch.randn(2, 3, 64, 64)}


@pytest.fixture
def mock_vae():
    vae = Mock(spec=AutoencoderKL)
    posterior = Mock()
    posterior.sample.return_value = torch.randn(2, 4, 8, 8)
    posterior.kl.return_value = torch.tensor([0.1, 0.1])
    vae.encode.return_value.latent_dist = posterior
    vae.decode.return_value.sample = torch.randn(2, 3, 64, 64)
    return vae


def test_extract_patches():
    # Test patch extraction
    image = torch.randn(2, 3, 64, 64)
    patch_size = 32
    stride = 16

    patches = extract_patches(image, patch_size, stride)

    # Calculate expected shape
    n_patches_h = (64 - patch_size) // stride + 1
    n_patches_w = (64 - patch_size) // stride + 1
    expected_shape = (2, 3, n_patches_h * n_patches_w, patch_size, patch_size)

    assert patches.shape == expected_shape


def test_patch_based_mse_loss():
    real_images = torch.randn(2, 3, 64, 64)
    recon_images = torch.randn(2, 3, 64, 64)

    loss = patch_based_mse_loss(real_images, recon_images, patch_size=32, stride=16)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)


@patch("lpips.LPIPS")
def test_train_step(mock_lpips, mock_args, mock_batch, mock_vae):
    accelerator = Accelerator()

    # Mock LPIPS loss
    mock_lpips_fn = Mock()
    mock_lpips_fn.return_value = torch.tensor([0.1])

    train_loss = 0.0

    loss, meta_loss = train_step(
        mock_args, mock_lpips_fn, mock_batch, accelerator, mock_vae, train_loss
    )

    assert isinstance(loss, torch.Tensor)
    assert isinstance(meta_loss, dict)
    assert "mse" in meta_loss
    assert "lpips" in meta_loss
    assert "kl" in meta_loss


@patch("datasets.load_dataset")
def test_prepare_dataset(mock_load_dataset, mock_args):
    accelerator = Accelerator()

    # Mock dataset
    mock_dataset = {
        "train": Mock(
            column_names=["image"],
            train_test_split=lambda: Mock(
                train=Mock(
                    with_transform=lambda: [
                        {"pixel_values": torch.randn(3, 64, 64)} for _ in range(5)
                    ]
                ),
                test=Mock(
                    with_transform=lambda: [
                        {"pixel_values": torch.randn(3, 64, 64)} for _ in range(2)
                    ]
                ),
            ),
        )
    }
    mock_load_dataset.return_value = mock_dataset

    train_dataloader, test_dataloader, train_len, test_len = prepare_dataset(
        mock_args, accelerator
    )

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(test_dataloader, DataLoader)
    assert train_len > 0
    assert test_len > 0


def test_patch_based_lpips_loss(mock_args):
    mock_lpips_model = Mock()
    mock_lpips_model.return_value = torch.tensor([0.1])

    real_images = torch.randn(2, 3, 64, 64)
    recon_images = torch.randn(2, 3, 64, 64)

    loss = patch_based_lpips_loss(
        mock_lpips_model, real_images, recon_images, patch_size=32, stride=16
    )

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


@pytest.mark.parametrize("train_only_decoder", [True, False])
def test_train_step_decoder_only(train_only_decoder, mock_args, mock_batch, mock_vae):
    accelerator = Accelerator()
    mock_args.train_only_decoder = train_only_decoder
    mock_lpips_fn = Mock()
    mock_lpips_fn.return_value = torch.tensor([0.1])

    train_loss = 0.0

    loss, meta_loss = train_step(
        mock_args, mock_lpips_fn, mock_batch, accelerator, mock_vae, train_loss
    )

    assert isinstance(loss, torch.Tensor)
    assert isinstance(meta_loss, dict)
    # In decoder-only mode, KL loss should not contribute to the total loss
    if train_only_decoder:
        assert (
            abs(loss - (meta_loss["mse"] + mock_args.lpips_scale * meta_loss["lpips"]))
            < 1e-6
        )
