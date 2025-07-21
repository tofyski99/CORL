# CORL: Causal Offline Reinforcement Learning for Bandwidth Estimation

This repository contains the code for the CORL project, which uses Causal Offline Reinforcement Learning for bandwidth estimation. The project is designed to train a model that can predict network bandwidth, evaluate its performance, and export the trained model to ONNX format for deployment.

## Prerequisites

Before you begin, ensure you have Python 3.8 or higher installed. The required Python packages are listed in `requirements.txt`.

You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

This will install the following packages:
- `matplotlib`
- `numpy`
- `onnxruntime`
- `pandas`
- `pyrallis`
- `scikit-learn`
- `torch`
- `tqdm`

## Usage

### Dataset

The project requires a specific directory structure for the datasets.

-   **Training Data**: The training script loads data from a pickle file located in the `training_dataset_pickle/` directory. You should place your training data pickle file here. The default path in the latest script is `training_dataset_pickle/v18_rand_20Per.pickle`.
-   **Evaluation Data**: The evaluation data should be placed under the `ALLdatasets/evaluate/` directory.
-   **Emulated Data**: Emulated datasets are expected in `ALLdatasets/EMU/`.


### Training the Model

To train the IQL model, you can run the `train.py` script from the `code` directory.

```bash
python code/train.py
```

The training script uses `pyrallis` for configuration management. You can modify the training parameters directly in the `TrainConfig` class within the script or override them via command-line arguments.

For example, to change the batch size and learning rate:
```bash
python code/train.py --batch_size=1024 --actor_lr=1e-4
```

Checkpoints for the trained models will be saved in the `checkpoints_iql/` directory.

### Evaluation

The script includes functionality to evaluate the performance of the trained model. The evaluation process is integrated into the training loop and can also be run separately. The `evaluate` function in the script handles the evaluation against the dataset in `ALLdatasets/evaluate/`.

### ONNX Export

The project supports exporting the trained PyTorch model to the ONNX format for cross-platform inference. The `export2onnx` function in the script handles this conversion. You will need to provide the path to the saved PyTorch model (`.pt` file) and the desired output path for the ONNX model.





