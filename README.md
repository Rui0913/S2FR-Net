# S<sup>2</sup>FR-Net

Official PyTorch implementation of **S<sup>2</sup>FR-Net: Semantic Reliability-Aware Spatial-Frequency Redistribution for High-Fidelity Pansharpening**, published in *IEEE Transactions on Geoscience and Remote Sensing* (TGRS), 2026.

**Authors:** Rui Hao, Xin Jin, Hongyue Huang, Keqin Li, Cheng Xie, and Qian Jiang

**Paper:** [https://doi.org/10.1109/TGRS.2026.3708931](https://doi.org/10.1109/TGRS.2026.3708931)

## Overview

Pansharpening reconstructs a high-resolution multispectral (HRMS) image from a high-resolution panchromatic (PAN) image and a low-resolution multispectral (LRMS) image. S<sup>2</sup>FR-Net treats this task as an **estimate-propagate-verify** process so that PAN details are transferred only when they are reliable for the underlying multispectral content.

The framework contains three coordinated components:

- **Frequency Redistribution Module (FRM):** estimates local transfer reliability in the wavelet domain and selectively redistributes directional high-frequency details.
- **Semantic Decision Propagation (SDP):** converts local reliability into a spatial-spectral fusion policy and propagates it through gated global interaction.
- **Cross-Modal Context Modeling (CCM):** verifies propagated responses against PAN structure and local fusion evidence before reconstruction.

Training uses the composite objective

```text
L = L_rec + 0.1 * L_sam + 0.3 * L_wav
```

where `L_rec` is the image-domain L1 loss, `L_sam` preserves spectral consistency, and `L_wav` supervises the Haar high-frequency subbands.

## Repository Structure

| File | Description |
| --- | --- |
| `main.py` | Training, validation, checkpointing, and loss definitions |
| `config.py` | Command-line configuration |
| `data.py` | Aligned PAN/MS/label dataset loader |
| `model.py` | Top-level pansharpening network and Haar transforms |
| `clip_rwkv_fusion.py` | CLIP-guided semantic decision propagation |
| `mamba_fusion.py` | Mamba-based cross-modal feature interaction |
| `context_attention.py` | Cross-modal context verification |
| `layers.py` | Shared convolutional and residual blocks |
| `refinement.py` | Feature refinement modules |

## Requirements

- Python 3.10 or newer
- PyTorch and TorchVision with a CUDA build compatible with your system
- OpenCV (`opencv-python`)
- NumPy
- Einops
- Transformers
- Mamba SSM and its CUDA dependencies

The experiments reported in the paper were run on a single NVIDIA GeForce RTX 3090 GPU. A CUDA-capable environment is strongly recommended.

Create an environment and install the dependencies as follows:

```bash
conda create -n s2frnet python=3.10 -y
conda activate s2frnet

# Install PyTorch and TorchVision for your CUDA version first:
# https://pytorch.org/get-started/locally/

pip install numpy opencv-python einops transformers
pip install causal-conv1d mamba-ssm
```

> [!IMPORTANT]
> `mamba_fusion.py` calls `Mamba(..., bimamba_type="v3")`. Use a Mamba SSM build that supports this argument. Exact package versions from the original experimental environment were not preserved, so this repository does not claim a fully pinned environment.

On the first run, Transformers downloads `openai/clip-vit-base-patch32` from Hugging Face. Network access is therefore required unless the model is already available in the local Hugging Face cache.

## Dataset Preparation

The code expects aligned training, validation, and test splits. Each split must contain `pan`, `ms`, and `label` directories with matching filenames:

```text
datasets/WV-2/
|-- train/
|   |-- pan/
|   |   `-- 0001.png
|   |-- ms/
|   |   `-- 0001.png
|   `-- label/
|       `-- 0001.png
|-- val/
|   |-- pan/
|   |-- ms/
|   `-- label/
`-- test/
    |-- pan/
    |-- ms/
    `-- label/
```

Expected image shapes are:

| Input | Shape | OpenCV loading mode |
| --- | --- | --- |
| PAN | `256 x 256 x 1` | Grayscale |
| LRMS | `64 x 64 x 3` | Color |
| HRMS label | `256 x 256 x 3` | Color |

Images are converted to tensors and normalized from `[0, 1]` to `[-1, 1]`. Files with the same name must exist in all three directories. The WorldView-II, QuickBird, and Maryland datasets used in the paper are not redistributed by this repository; obtain and prepare them in accordance with their respective licenses.

## Training

The following command explicitly reproduces the main optimization settings reported in the paper: batch size 16, 8000 epochs, peak learning rate `2e-4`, and validation every 50 epochs.

```bash
python main.py \
  --dataset-train /path/to/datasets/WV-2/train \
  --dataset-val /path/to/datasets/WV-2/val \
  --dataset-test /path/to/datasets/WV-2/test \
  --batch-size 16 \
  --epochs 8000 \
  --learning-rate 2e-4 \
  --validation-frequency 50 \
  --cuda \
  --gpu-id 0 \
  --network-name S2FR-Net
```

`--dataset-test` is retained by the command-line interface but is not consumed by the current training entry point. The repository currently provides training and validation only; it does not yet include a supported standalone inference or metric-evaluation command.

The learning rate is warmed up linearly during the first 50 epochs, held at `2e-4` through epoch 6000, and reduced to `1e-4` for the remaining epochs. The optimizer is Adam with betas `(0.9, 0.999)`.

### Resume Training

Append the following option to the training command to restore the model, optimizer, and epoch from a checkpoint produced by the current code:

```bash
--resume /path/to/latest.pth
```

To initialize only the model weights from a compatible state dict, use:

```bash
--pretrained /path/to/compatible_weights.pth
```

Do not pass `--resume` and `--pretrained` together; resume takes precedence.

## Outputs

By default, each run is stored under:

```text
logs/<timestamp>_batch_size_<batch-size>_lr_<learning-rate>_WV2/
|-- backup_dir/
|   `-- latest.pth
|-- bestmodel_dir/
|   `-- model_best_epoch.pth
|-- train_image/
|-- val_image/
|-- ori_image/
|-- <network-name>_<timestamp>_train.log
|-- <network-name>_<timestamp>_validation.log
|-- <network-name>_<timestamp>_epoch_time.log
`-- <network-name>_<timestamp>_best_epoch.log
```

`latest.pth` and `model_best_epoch.pth` contain model and optimizer state generated by the current training code. Training comparison images are saved every 100 epochs, while a validation comparison image is saved at each validation interval.

## Pretrained Weights

Compatible pretrained weights are not currently published. A local 611 MB legacy checkpoint was intentionally excluded because it serializes the old Python model object, references retired module paths and `cuda:3`, and cannot be loaded reliably by the refactored code in this repository. A converted and validated WorldView-II checkpoint may be released separately in the future.

## Citation

If this work is useful in your research, please cite:

```bibtex
@article{hao2026s2frnet,
  author  = {Rui Hao and Xin Jin and Hongyue Huang and Keqin Li and Cheng Xie and Qian Jiang},
  title   = {S$^2$FR-Net: Semantic Reliability-Aware Spatial--Frequency Redistribution for High-Fidelity Pansharpening},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {64},
  pages   = {1--17},
  year    = {2026},
  note    = {Art. no. 5406517},
  doi     = {10.1109/TGRS.2026.3708931}
}
```

## License

This project is released under the [MIT License](LICENSE).
