
# Extreme Multi-Label Text Classification with DistilBERT Integrated LightXML

This repository contains experiments using LightXML integrated with DistilBERT for extreme multi-label text classification tasks across three datasets: EURLEX-4K, WIKI-10-31K, and AMAZONCAT-13K.

## Prerequisites

To run this project, make sure you have the following set up:

- Docker installed on your system
- NVIDIA drivers with CUDA 11.7 properly configured

## Steps to Run the Project

### 1. Build the Docker Image

Create a `Dockerfile` with the following content:

```Dockerfile
# Base image for Ubuntu 20.04 with CUDA 11.7 support
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Install essential tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    curl \
    git
```

### 2. Run the Docker Container

To build and run the Docker container and access the bash CLI, use the following commands:

```bash
docker build -t lightxml-env .
docker run --gpus all -it lightxml-env /bin/bash
```

### 3. Clone the LightXML Repository

Once inside the container, clone the [LightXML repository](https://github.com/kongds/LightXML) and install the required packages:

```bash
git clone https://github.com/kongds/LightXML.git
cd LightXML
pip install -r requirements.txt
```

### 4. Install PyTorch and CUDA

Make sure PyTorch is installed with CUDA 11.7 support:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

### 5. Download and Unzip the Datasets

Download and unzip the datasets into the `data` directory as per the LightXML repository's instructions. Ensure the datasets are organized correctly.

### 6. Replace Files in `/src`

Replace the necessary files in the `/src` directory with the ones provided along with this README to ensure the latest changes are reflected in the code.

### 7. Run the Experiments

Use the following commands to run the experiments and produce results similar to those in the dissertation:

#### EURLEX-4K Dataset:

```bash
python3 src/main.py --lr 1e-4 --epoch 20 --dataset eurlex4k --swa --swa_warmup 10 --swa_step 400 --batch 8 --update_count 2 --bert xlnet
python3 src/main.py --lr 1e-4 --epoch 20 --dataset eurlex4k --swa --swa_warmup 10 --swa_step 400 --batch 8 --update_count 2 --bert distilbert
python3 src/main.py --lr 1e-4 --epoch 20 --dataset eurlex4k --swa --swa_warmup 10 --swa_step 200 --batch 16 --bert distilbert
python3 src/main.py --lr 1e-4 --epoch 20 --dataset eurlex4k --swa --swa_warmup 10 --swa_step 200 --batch 16
python3 src/main.py --lr 1e-4 --epoch 20 --dataset eurlex4k --swa --swa_warmup 10 --swa_step 200 --batch 16 --bert roberta
```

#### WIKI-10-31K Dataset:

```bash
python3 src/main.py --lr 1e-4 --epoch 30 --dataset wiki31k --swa --swa_warmup 10 --swa_step 300 --batch 16
python3 src/main.py --lr 1e-4 --epoch 30 --dataset wiki31k --swa --swa_warmup 10 --swa_step 300 --batch 16 --bert distilbert
python3 src/main.py --lr 1e-4 --epoch 30 --dataset wiki31k --swa --swa_warmup 10 --swa_step 400 --batch 8 --update_count 2 --bert xlnet
python3 src/main.py --lr 1e-4 --epoch 30 --dataset wiki31k --swa --swa_warmup 10 --swa_step 400 --batch 8 --update_count 2 --bert roberta
python3 src/main.py --lr 1e-4 --epoch 30 --dataset wiki31k --swa --swa_warmup 10 --swa_step 400 --batch 8 --update_count 2 --bert distilbert
```

#### AMAZONCAT-13K Dataset:

```bash
python3 src/main.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 2 --swa_step 10000 --batch 16 --eval_step 20000
python3 src/main.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 2 --swa_step 10000 --batch 16 --eval_step 20000 --bert roberta
python3 src/main.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 2 --swa_step 10000 --batch 16 --eval_step 20000 --bert distilbert
python3 src/main.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 2 --swa_step 10000 --batch 32 --eval_step 20000 --bert xlnet --max_len 128
python3 src/main.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 2 --swa_step 10000 --batch 32 --eval_step 20000 --bert distilbert
```

### 8. Run Models on Test Datasets

After training, use the following commands to run the models on the test datasets:

#### EURLEX-4K Dataset:

```bash
python3 src/modelinference.py --dataset eurlex4k
```

#### WIKI-10-31K Dataset:

```bash
python3 src/modelinference.py --dataset wiki31k
```

#### AMAZONCAT-13K Dataset:

```bash
python3 src/modelinference.py --dataset amazoncat13k
```

## Additional Notes

- Ensure that the versions of PyTorch and CUDA are compatible with your system. This setup was tested on CUDA 11.7 and PyTorch 2.0.1.
- For more details on dataset organization, model specifics, and further instructions, refer to the official [LightXML repository](https://github.com/kongds/LightXML).
