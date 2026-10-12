# Compact CNN for CIFAR-10 — TinyML

## Project Overview

This project implements a Compact Convolutional Neural Network (CCNN) for CIFAR-10 image classification, focusing on model accuracy, memory efficiency, and CPU inference performance.

The architecture combines depthwise separable convolutions, Squeeze-and-Excitation blocks, feature compression and Global Average Pooling.

The trained PyTorch model is exported to ONNX and evaluated using static and dynamic INT8 quantization.

## Requirements

- Python 3.12 or 3.13 with compatible package versions
- Google Colab with GPU recommended for training
- Packages listed in `Marcos_Dominguez_requirements.txt`

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install the required packages:

```bash
python -m pip install -r Marcos_Dominguez_requirements.txt
```

## Running the Notebook

1. Open `Marcos_Dominguez.ipynb` in Google Colab or VS Code with Jupyter support.
2. Select a Python environment with the required packages installed.
3. Run all cells sequentially.
4. CIFAR-10 will be downloaded automatically into the `data/` directory.
5. If trained weights are unavailable, the CCNN will train from scratch for 100 epochs.
6. The notebook exports FP32 ONNX and generates static and dynamic INT8 models.
7. Results and generated models are saved in the `artifacts/` directory.

No Google Drive connection or manually downloaded dataset is required.

## Model Configuration

- Dataset: CIFAR-10 (45,000 training, 5,000 validation, 10,000 test images)
- Architecture: Compact CNN with five depthwise separable blocks and Squeeze-and-Excitation
- Trainable parameters: 108,178
- Training: 100 epochs, batch size 128
- Optimizer: SGD with momentum and Nesterov acceleration
- Scheduler: OneCycleLR
- Static INT8 calibration: 500 training images

## Results

| Model | Size (KiB) | Test Accuracy | CPU Latency (ms/image) |
|---|---:|---:|---:|
| PyTorch FP32 (estimated footprint) | 431.6 | 89.88% | 3.82 |
| ONNX FP32 | 429.2 | 89.88% | 0.63 |
| ONNX Static INT8 | 186.3 | 89.58% | 0.50 |
| ONNX Dynamic INT8 (Conv + Gemm) | 159.1 | 89.78% | 1.85 |
| ONNX Dynamic INT8 (Linear Only) | 430.5 | 89.90% | 0.42 |

Static INT8 provides a practical balance between model size, accuracy, and CPU inference latency. The dynamic Conv + Gemm configuration produces a smaller model but has higher measured CPU latency.

## Limitations

Inference latency was measured using ONNX Runtime on a CPU with batch size 1 and one inference thread. These results do not establish actual latency, peak RAM usage or an operator compatibility on a microcontroller.

Small differences in accuracy and latency may occur depending on the hardware, package versions, and execution environment.