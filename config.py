"""Command-line configuration for model training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the CLIP-RWKV pan-sharpening model."
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--dataset-train",
        "--dataset_train",
        dest="dataset_train",
        default="../../Second_work/test/datasets/WV-2/train",
        help="Training dataset root containing pan, ms, and label folders.",
    )
    data_group.add_argument(
        "--dataset-val",
        "--dataset_val",
        dest="dataset_val",
        default="../../Second_work/test/datasets/WV-2/val",
        help="Validation dataset root containing pan, ms, and label folders.",
    )
    data_group.add_argument(
        "--dataset-test",
        "--dataset_test",
        dest="dataset_test",
        default="../../Second_work/test/datasets/WV-2/test",
        help="Test dataset root containing pan, ms, and label folders.",
    )
    data_group.add_argument(
        "--batch-size",
        "--batchSize",
        dest="batch_size",
        type=int,
        default=32,
        help="Number of samples per training batch.",
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument(
        "--learning-rate", "--lr", dest="learning_rate", type=float, default=0.0002
    )
    training_group.add_argument(
        "--start-epoch", "--start_epoch", dest="start_epoch", type=int, default=1
    )
    training_group.add_argument(
        "--epochs", "--nEpochs", dest="epochs", type=int, default=11000
    )
    training_group.add_argument(
        "--validation-frequency",
        "--val_freq",
        dest="validation_frequency",
        type=int,
        default=50,
    )
    training_group.add_argument("--seed", type=int, default=123)
    training_group.add_argument(
        "--resume", default="", help="Checkpoint used to resume training."
    )
    training_group.add_argument(
        "--pretrained", default="", help="Pretrained model or state-dict path."
    )

    device_group = parser.add_argument_group("device")
    cuda_group = device_group.add_mutually_exclusive_group()
    cuda_group.add_argument(
        "--cuda", dest="cuda", action="store_true", help="Use CUDA (default)."
    )
    cuda_group.add_argument(
        "--no-cuda", dest="cuda", action="store_false", help="Run on CPU."
    )
    device_group.set_defaults(cuda=True)
    device_group.add_argument(
        "--gpu-id", "--gpu_id", dest="gpu_id", type=int, default=0
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--network-name", "--net", dest="network_name", default="Demo"
    )
    output_group.add_argument(
        "--logs", default="logs", help="Root directory for run outputs."
    )
    output_group.add_argument(
        "--best-model", "--best_model", dest="best_model", default="bestmodel_dir"
    )
    output_group.add_argument("--backup", default="backup_dir")
    output_group.add_argument(
        "--train-path", "--train_path", dest="train_path", default="train_image"
    )
    output_group.add_argument(
        "--val-path", "--val_path", dest="val_path", default="val_image"
    )
    output_group.add_argument(
        "--ori-path", "--ori_path", dest="ori_path", default="ori_image"
    )
    output_group.add_argument(
        "--train-log", "--train_log", dest="train_log", default=""
    )
    output_group.add_argument(
        "--epoch-time-log", "--epoch_time_log", dest="epoch_time_log", default=""
    )
    output_group.add_argument(
        "--best-epoch-log", "--best_epoch_log", dest="best_epoch_log", default=""
    )
    output_group.add_argument("--val-log", "--val_log", dest="val_log", default="")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments without doing work at import time."""
    return build_parser().parse_args(argv)
