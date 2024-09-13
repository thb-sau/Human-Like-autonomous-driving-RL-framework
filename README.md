# Human-Like-autonomous-driving-RL-framework
RL autonomous driving algorithm built using PyTorch framework. Run and SMART simulation platform.

## Environment Setup

### Step 1: Create a Conda Environment
Firstly, you need to set up a new conda environment with Python version 3.9 or higher.

```bash
conda create -n HLAD python=3.9
conda activate HLAD
```

### Step 2: Install SMARTS Simulator
Secondly, install the [SMARTS](https://github.com/huawei-noah/SMARTS "SMARTS") simulator, ensuring it is compatible with version 2.0.x. Follow the installation instructions provided in the official repository.

### Step 3: Install Dependencies
Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/thb-sau/Human-Like-autonomous-driving-RL-framework
cd Human-Like-autonomous-driving-RL-framework
```

Install the necessary dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Training

### Step 1: Collect Expert Data
Collect expert driving data which will be saved under the `data` folder.

### Step 2: Train ViT Autoencoder
Utilize the collected expert data to train the Vision Transformer (ViT) Autoencoder. Replace `[data_name]` with the filename of the expert data file, for example, `expert.hdf5`.

```bash
python train_encoder/run.py --data_name [data_name]
```

### Step 3: Train Policy Network
Proceed to train the policy network using the following command. Replace `[path]` with the path to the trained encoder model and `[name]` with the name of the expert data file.

```bash
python train.py --encoder_model_path [path] --expert_data_name [name]
```

For training other algorithms, refer to their respective scripts within the `algorithm` directory:

```bash
python algorithm/SAC/trainer.py --encoder_model_path [path]
```

## Evaluation

To evaluate the performance of unprotected left turns, use the following command. Replace `[path]` with the path to the trained encoder model, `[dir_path]` with the directory where the policy mode is stored, and `[log_dir]` with the directory where logs should be written.

```bash
python vehicle_condition.py --env_mode left_t --encoder_model_path [path] --policy_mode_dir [dir_path] --log_dir [log_dir]
```

This document provides a basic guide to setting up your environment, training your models, and evaluating their performance within the Human-Like Autonomous Driving Reinforcement Learning Framework. For more detailed instructions and troubleshooting, please refer to the documentation and examples provided in the repository.