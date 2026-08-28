# Autoencoder Coursework

A PyTorch-based image autoencoder project for reconstructing RGB images and exploring the effect of training hyperparameters. The repository also contains an early classification experiment using flattened image features.

## Project Contents

- `Autoencoder .ipynb` - Loads and preprocesses the image subsets, trains the convolutional autoencoder, evaluates reconstruction error, and plots loss curves and reconstructed images.
- `demo_code_classification.ipynb` - Prototype classification workflow using image features and one-hot encoded labels.
- `subset_1.npy`, `subset_2.npy`, `subset_3.npy` - Image data subsets used by the autoencoder notebook.
- `*_mse_losses.npy` - Saved loss histories for selected batch sizes and learning rates.
- `Coursework 2 Brief.pdf` - Coursework requirements.
- `Sample Report Structure.docx` - Suggested report structure.

## Model

The active autoencoder uses three convolutional encoder layers and three transpose-convolution decoder layers:

- Input: RGB image with shape `(3, 150, 225)`
- Latent representation: `(32, 10, 15)`
- Decoder output: reconstructed RGB image with shape `(3, 150, 225)`
- Activation: ReLU in hidden layers and Sigmoid at the output
- Reconstruction loss: mean squared error (MSE)
- Optimizer: Adam

The default training configuration is:

- Batch size: `32`
- Learning rate: `0.01`
- Epochs: `30`
- Device: CUDA when available, otherwise CPU

## Requirements

Python 3.9 or newer is recommended. Install the required packages with:

```bash
python -m pip install numpy matplotlib torch scikit-learn jupyter
```

A GPU is optional. The notebook automatically selects CUDA when PyTorch detects it.

## Running the Autoencoder

1. Open `Autoencoder .ipynb` in JupyterLab or VS Code with the Jupyter extension.
2. Select a Python environment containing the dependencies above.
3. Run the cells from top to bottom.
4. Review the training MSE curve, test loss, and original/reconstructed image plots.

The notebook combines the three `.npy` subsets, normalizes `uint8` pixel values to `[0, 1]`, and uses an 80/20 train/test split.

## Loss Comparisons

The saved loss files can be loaded with NumPy to compare experiments:

```python
import numpy as np

batch_16 = np.load("16_mse_losses.npy")
batch_32 = np.load("32_mse_losses.npy")
batch_64 = np.load("64_mse_losses.npy")

learning_rate_001 = np.load("0.001_mse_losses.npy")
learning_rate_01 = np.load("0.01_mse_losses.npy")
learning_rate_1 = np.load("0.1_mse_losses.npy")
```

## Classification Demo Status

`demo_code_classification.ipynb` is a prototype and is not currently runnable from the repository as-is. It expects `features.npy` and `labels.npy`, which are not included in the current workspace. It also contains unfinished implementation details that may need correction before use, including the classifier module registration and optimizer/training setup.

## Notes

- The notebooks have not been committed with executed outputs, so results are generated when the cells are run.
- The autoencoder notebook includes commented alternative encoder/decoder sizes for comparing compression ratios.
- The `.npy` datasets must remain in the same directory as the notebooks unless the loading paths are updated.
