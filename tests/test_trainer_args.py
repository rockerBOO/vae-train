import pytest
from vae_train.trainer_args import parse_args, TrainerArgs
import sys
from unittest.mock import patch


def test_trainer_args_default_values():
    # Test with minimum required args
    test_args = ["prog"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(
            ValueError, match="Need either a dataset name or a training folder."
        ):
            parse_args()


def test_trainer_args_with_dataset_name():
    # Test with dataset_name provided
    test_args = ["prog", "--dataset_name", "test_dataset"]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert isinstance(args, TrainerArgs)
        assert args.dataset_name == "test_dataset"
        assert args.train_data_dir is None
        assert args.train_batch_size == 1
        assert args.learning_rate == 1.5e-7
        assert args.resolution == 512


def test_trainer_args_with_train_data_dir():
    # Test with train_data_dir provided
    test_args = ["prog", "--train_data_dir", "/path/to/data"]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert isinstance(args, TrainerArgs)
        assert args.train_data_dir == "/path/to/data"
        assert args.dataset_name is None


def test_trainer_args_with_custom_values():
    # Test with multiple custom arguments
    test_args = [
        "prog",
        "--dataset_name",
        "custom_dataset",
        "--train_batch_size",
        "32",
        "--learning_rate",
        "0.001",
        "--resolution",
        "256",
        "--seed",
        "42",
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--patch_loss",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert isinstance(args, TrainerArgs)
        assert args.dataset_name == "custom_dataset"
        assert args.train_batch_size == 32
        assert args.learning_rate == 0.001
        assert args.resolution == 256
        assert args.seed == 42
        assert args.gradient_checkpointing is True
        assert args.use_8bit_adam is True
        assert args.patch_loss is True


def test_trainer_args_mixed_precision_validation():
    # Test mixed precision argument validation
    test_args = [
        "prog",
        "--dataset_name",
        "test_dataset",
        "--mixed_precision",
        "invalid_value",
    ]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit):
            parse_args()


def test_trainer_args_lr_scheduler_values():
    # Test valid lr_scheduler values
    valid_schedulers = [
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ]

    for scheduler in valid_schedulers:
        test_args = [
            "prog",
            "--dataset_name",
            "test_dataset",
            "--lr_scheduler",
            scheduler,
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            assert args.lr_scheduler == scheduler


def test_trainer_args_scale_lr_default():
    test_args = ["prog", "--dataset_name", "test_dataset"]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.scale_lr is True  # According to default in argparse setup


def test_trainer_args_dataclass_fields():
    # Test that all required fields are present in TrainerArgs dataclass
    required_fields = {
        "pretrained_model_name_or_path",
        "dataset_name",
        "output_dir",
        "learning_rate",
        "train_batch_size",
        "num_train_epochs",
    }

    trainer_args_fields = {
        field.name for field in TrainerArgs.__dataclass_fields__.values()
    }
    assert required_fields.issubset(trainer_args_fields)
