"""Training entry point for the CLIP-RWKV pan-sharpening model."""

from __future__ import annotations

import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import parse_args
from data import PanSharpeningDataset
from model import HaarDWT, PanSharpeningNet


class S2FRCompositeLoss(nn.Module):
    """Composite loss from Eqs. (19)-(22) of the S2FR-Net manuscript.

    L = L_rec + lambda_sam * L_sam + lambda_wav * L_wav

    where L_rec is the image-domain L1 reconstruction loss, L_sam is the
    mean spectral-angle loss, and L_wav is the sum of L1 losses on the
    level-1 Haar HL/LH/HH high-frequency subbands.
    """

    def __init__(
        self,
        lambda_sam: float = 0.1,
        lambda_wav: float = 0.3,
        eps: float = 1e-8,
        acos_margin: float = 1e-7,
    ) -> None:
        super().__init__()
        if lambda_sam < 0 or lambda_wav < 0:
            raise ValueError("Loss weights must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be greater than zero.")
        if not 0 < acos_margin < 1:
            raise ValueError("acos_margin must be in (0, 1).")

        self.lambda_sam = float(lambda_sam)
        self.lambda_wav = float(lambda_wav)
        self.eps = float(eps)
        self.acos_margin = float(acos_margin)
        # Reuse exactly the same level-1 Haar definition as the network.
        self.dwt = HaarDWT()

    def reconstruction_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return functional.l1_loss(prediction, target, reduction="mean")

    def spectral_angle_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError(
                "SAM expects prediction and target with shape [B, C, H, W]."
            )

        # Per-pixel spectral vectors: [B, C, H, W] -> angle over C.
        dot_product = (prediction * target).sum(dim=1)
        prediction_norm = torch.linalg.vector_norm(prediction, ord=2, dim=1)
        target_norm = torch.linalg.vector_norm(target, ord=2, dim=1)
        denominator = prediction_norm * target_norm + self.eps
        cosine = dot_product / denominator

        # acos has an unbounded derivative at +/-1.  The manuscript includes
        # epsilon for numerical stability; this clamp additionally prevents
        # round-off from creating NaNs while preserving the SAM definition.
        cosine = cosine.clamp(
            min=-1.0 + self.acos_margin,
            max=1.0 - self.acos_margin,
        )
        return torch.acos(cosine).mean()

    def wavelet_high_frequency_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        _, pred_hl, pred_lh, pred_hh = self.dwt(prediction)
        _, target_hl, target_lh, target_hh = self.dwt(target)
        return (
            functional.l1_loss(pred_hl, target_hl, reduction="mean")
            + functional.l1_loss(pred_lh, target_lh, reduction="mean")
            + functional.l1_loss(pred_hh, target_hh, reduction="mean")
        )

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target must have the same shape, got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if prediction.ndim != 4:
            raise ValueError(
                f"Expected [B, C, H, W] tensors, got {tuple(prediction.shape)}."
            )
        if prediction.shape[-2] % 2 != 0 or prediction.shape[-1] % 2 != 0:
            raise ValueError(
                "The Haar wavelet loss requires even spatial dimensions."
            )

        loss_rec = self.reconstruction_loss(prediction, target)
        loss_sam = self.spectral_angle_loss(prediction, target)
        loss_wav = self.wavelet_high_frequency_loss(prediction, target)
        return loss_rec + self.lambda_sam * loss_sam + self.lambda_wav * loss_wav


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    best_models: Path
    train_images: Path
    validation_images: Path
    original_images: Path


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(use_cuda: bool, gpu_id: int) -> torch.device:
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA-capable GPU is available.")
    return torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")


def create_run_paths(args, timestamp: str) -> RunPaths:
    run_name = (
        f"{timestamp}_batch_size_{args.batch_size}_"
        f"lr_{args.learning_rate:.6f}_WV2"
    )
    root = Path(args.logs) / run_name
    paths = RunPaths(
        root=root,
        checkpoints=root / args.backup,
        best_models=root / args.best_model,
        train_images=root / args.train_path,
        validation_images=root / args.val_path,
        original_images=root / args.ori_path,
    )
    for path in paths.__dict__.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def open_log(
    stack: ExitStack,
    configured_path: str,
    default_path: Path,
) -> TextIO:
    path = Path(configured_path) if configured_path else default_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return stack.enter_context(path.open("a" if configured_path else "w", encoding="utf-8"))


def load_model_state(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        stored_model = checkpoint["model"]
        state_dict = (
            stored_model.state_dict() if isinstance(stored_model, nn.Module) else stored_model
        )
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def resume_training(
    model: nn.Module,
    optimizer: Adam,
    checkpoint_path: str,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Resume checkpoint must contain model_state_dict.")
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)) + 1


def learning_rate_for_epoch(epoch: int, peak_learning_rate: float) -> float:
    if epoch <= 50:
        return max(peak_learning_rate * epoch / 50, 1e-6)
    if epoch <= 6000:
        return peak_learning_rate
    return min(peak_learning_rate, 1e-4)


def save_comparison_image(
    output_dir: Path,
    prediction: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
) -> None:
    comparison = torch.cat([prediction[0], target[0]], dim=2).clamp(-1, 1)
    image = ((comparison * 0.5 + 0.5) * 255).byte()
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    cv2.imwrite(str(output_dir / f"{epoch}_output.png"), image)


def train_one_epoch(
    data_loader: DataLoader,
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: Adam,
    device: torch.device,
    epoch: int,
    image_dir: Path,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_index, (pan, ms, target, _) in enumerate(data_loader):
        pan = pan.to(device, non_blocking=True)
        ms = ms.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(ms, pan)
        loss = loss_function(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = target.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if epoch % 100 == 0 and batch_index == 0:
            save_comparison_image(image_dir, output, target, epoch)

    if total_samples == 0:
        raise RuntimeError("The training data loader produced no samples.")
    return total_loss / total_samples


@torch.no_grad()
def validate(
    data_loader: DataLoader,
    model: nn.Module,
    loss_function: nn.Module,
    device: torch.device,
    epoch: int,
    image_dir: Path,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch_index, (pan, ms, target, _) in enumerate(data_loader):
        pan = pan.to(device, non_blocking=True)
        ms = ms.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(ms, pan)

        batch_size = target.shape[0]
        total_loss += loss_function(output, target).item() * batch_size
        total_samples += batch_size
        if batch_index == 0:
            save_comparison_image(image_dir, output, target, epoch)

    if total_samples == 0:
        raise RuntimeError("The validation data loader produced no samples.")
    return total_loss / total_samples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Adam,
    epoch: int,
    validation_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def run_training(args) -> None:
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if args.validation_frequency <= 0:
        raise ValueError("validation_frequency must be greater than zero.")

    device = select_device(args.cuda, args.gpu_id)
    set_random_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_dataset = PanSharpeningDataset(args.dataset_train)
    validation_dataset = PanSharpeningDataset(args.dataset_val)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = PanSharpeningNet(device=device).to(device)
    loss_function = S2FRCompositeLoss(lambda_sam=0.1, lambda_wav=0.3).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(
        "Loss: L_rec + 0.1 * L_sam + 0.3 * L_wav "
        "(L_wav uses Haar HL/LH/HH subbands)"
    )

    start_epoch = args.start_epoch
    if args.resume:
        start_epoch = resume_training(model, optimizer, args.resume, device)
        print(f"Resumed training from epoch {start_epoch}")
    elif args.pretrained:
        load_model_state(model, args.pretrained, device)
        print(f"Loaded pretrained weights from {args.pretrained}")

    timestamp = time.strftime("%Y%m%d%H%M")
    paths = create_run_paths(args, timestamp)
    best_validation_loss = float("inf")

    with ExitStack() as stack:
        train_log = open_log(
            stack,
            args.train_log,
            paths.root / f"{args.network_name}_{timestamp}_train.log",
        )
        epoch_time_log = open_log(
            stack,
            args.epoch_time_log,
            paths.root / f"{args.network_name}_{timestamp}_epoch_time.log",
        )
        best_epoch_log = open_log(
            stack,
            args.best_epoch_log,
            paths.root / f"{args.network_name}_{timestamp}_best_epoch.log",
        )
        validation_log = open_log(
            stack,
            args.val_log,
            paths.root / f"{args.network_name}_{timestamp}_validation.log",
        )

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            learning_rate = learning_rate_for_epoch(epoch, args.learning_rate)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            train_loss = train_one_epoch(
                train_loader,
                model,
                loss_function,
                optimizer,
                device,
                epoch,
                paths.train_images,
            )
            message = (
                f"Epoch [{epoch}/{args.epochs}] "
                f"lr={learning_rate:.8f} train_loss={train_loss:.10f}"
            )
            print(message)
            train_log.write(message + "\n")
            train_log.flush()

            if epoch % args.validation_frequency == 0:
                validation_loss = validate(
                    validation_loader,
                    model,
                    loss_function,
                    device,
                    epoch,
                    paths.validation_images,
                )
                validation_log.write(f"{epoch} {validation_loss:.10f}\n")
                validation_log.flush()
                save_checkpoint(
                    paths.checkpoints / "latest.pth",
                    model,
                    optimizer,
                    epoch,
                    validation_loss,
                )

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    save_checkpoint(
                        paths.best_models / "model_best_epoch.pth",
                        model,
                        optimizer,
                        epoch,
                        validation_loss,
                    )
                    best_epoch_log.write(f"{epoch} {validation_loss:.10f}\n")
                    best_epoch_log.flush()

            elapsed_minutes = (time.time() - epoch_start) / 60
            epoch_time_log.write(f"{epoch} {elapsed_minutes:.4f}\n")
            epoch_time_log.flush()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
