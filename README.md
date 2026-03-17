# CG4002 Capstone - AI Accelerator

Multi-task learning neural network for a pickleball AI system, trained on a CPU/GPU and deployed to an Ultra96 FPGA platform for hardware-accelerated inference.

The HLS source code for the FPGA accelerator is maintained in a separate repository:
https://github.com/CG4002-AY2526S2-B03/CG4002_capstone_hls

---

## Overview

The model takes a 6D ball state (position + velocity) as input and simultaneously predicts:
- **Regression:** 6D racket target state (position + velocity)
- **Classification:** Shot type (Drive, Drop, Dink, Lob, SpeedUp, HandBattle)

The trained weights are quantized to INT8 and exported for HLS synthesis, then deployed to the Ultra96 via a PS-PL DMA interface.

---

## Repository Structure

### training/
Scripts for dataset generation, model training, and hyperparameter tuning.

| File | Description |
|------|-------------|
| `model.py` | MTL network architecture with shared trunk and two task heads |
| `generate_dataset.py` | Synthetic pickleball dataset generation using physics simulation |
| `prepare_dataset.py` | Data cleaning, normalization, and train/val/test splitting |
| `tuner.py` | Multi-objective Optuna hyperparameter search |
| `train.py` | Final training script using best hyperparameters |
| `generate_test_vectors.py` | Generates test inputs for FPGA evaluation |
| `export_weights_int8.py` | Exports INT8 quantized weights as C headers for HLS |

### artifacts/
Training outputs including model weights, performance plots, and Optuna results.

| Path | Description |
|------|-------------|
| `final_model/best_model.pth` | Trained PyTorch checkpoint |
| `final_model/model.onnx` | ONNX export for cross-platform inference |
| `final_model/train_report.json` | Per-epoch training statistics |
| `best_params.json` | Pareto-optimal hyperparameters from Optuna |
| `optuna_study.db` | SQLite database of all tuning trials |
| `scaler_params.json` | Input normalization parameters |
| `plots/` | Confusion matrices, F1 scores, regression errors, training curves |

### ultra96_deploy/
FPGA bitstream, hardware drivers, and inference code for the Ultra96 platform.

| File | Description |
|------|-------------|
| `design_1.bit` | FPGA bitstream with accelerated neural network |
| `design_1.hwh` | Hardware description (DMA address space, pin mapping) |
| `ps_dma_driver.py` | MMIO DMA driver for FPGA inference via PYNQ |
| `predict_fpga.py` | High-level inference wrapper |
| `eval_hw.py` | FPGA hardware evaluation (latency, accuracy) |
| `eval_sw.py` | CPU software baseline for comparison |
| `power_management.py` | CPU frequency and power monitoring |
| `fused_weights.npz` | Batch-norm fused weights for FPGA deployment |

### comms/
MQTT communication layer for the distributed pickleball system.

| File | Description |
|------|-------------|
| `u96_client.py` | MQTT client with TLS authentication |
| `u96_client_insecure.py` | Plain-text MQTT client for testing |
| `ai_u96_client.py` | AI-aware MQTT client: receives ball state, runs inference, publishes result |
| `ai_event_generator.py` | Synthetic ball event generator for testing |

---

## Model Details

- **Architecture:** Shared FC trunk (2 x 512 hidden units, BatchNorm, ReLU6, Dropout) with separate regression and classification heads
- **Input:** 6D ball state (x, y, z, vx, vy, vz)
- **Outputs:** 6D racket state (regression) + 6-class shot type (classification)
- **Loss:** Weighted L1 (regression) + cross-entropy (classification)
- **Quantization:** INT8 symmetric per-tensor
- **Best performance:** F1 = 0.942, normalized MAE = 0.078

---

## Pipeline

1. Generate synthetic dataset with physics simulation (`generate_dataset.py`)
2. Prepare and normalize data (`prepare_dataset.py`)
3. Run hyperparameter search (`tuner.py`)
4. Train final model (`train.py`)
5. Export INT8 weights for HLS (`export_weights_int8.py`)
6. Synthesize HLS and generate bitstream (see [CG4002_capstone_hls](https://github.com/CG4002-AY2526S2-B03/CG4002_capstone_hls))
7. Deploy and evaluate on Ultra96 (`eval_hw.py`, `predict_fpga.py`)
