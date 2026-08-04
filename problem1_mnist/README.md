# Problem 1: MNIST CNN

This project contains the code for training and evaluating a convolutional neural network on the MNIST dataset.

## Repository Structure

- `checkpoints/`: Saved model checkpoints.
- `runs/`: TensorBoard event files.
- `configs/`: YAML configuration files for model and data setup.
- `train_best_config.sh`: Script to train using the best configuration.
- `test_best_config.sh`: Script to evaluate the best checkpoint.
- `src/`: Source code for the project.

## Environment Setup

Create a new Conda environment:

```bash
conda create -n mnist_env python=3.10 -y
conda activate mnist_env
```

Install the required packages:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

If you are using a different CUDA version, update the `--extra-index-url` accordingly.

## Data Setup

Before training, make sure to update the dataset path in `configs/data.yaml`.

The MNIST data was downloaded locally and used from that path, so you should point the config file to your own data location.

## Reproducing the Best Results

The best checkpoint is saved under the `checkpoints/` folder.  
To reproduce the reported results, run:

```bash
chmod +x test_best_config.sh
./test_best_config.sh
```

## Training

To train the model using the best configuration, run:

```bash
chmod +x train_best_config.sh
./train_best_config.sh
```

Both `train_best_config.sh` and `test_best_config.sh` use the model and config files defined in the repository.

## TensorBoard

The best TensorBoard logs are saved under the `runs/` folder.  
You can inspect them with:

```bash
tensorboard --logdir runs
```

**Use this [google drive](https://drive.google.com/drive/folders/1pen5tLmVZ5IYvVNGEpWSxgi2f80F3onh?usp=sharing) link to download all the tensorboard data (folder name is  `runs`). Download it and place in `problem1_mnist` folder**

## Notes

- The best model checkpoint is available in `checkpoints/`.
- The best TensorBoard event file is available in `runs/`.
- Make sure the config paths are correct before running training or evaluation.