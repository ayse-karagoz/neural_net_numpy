# Neural Network from Scratch with NumPy

A simple feedforward neural network implementation built from scratch using Python and NumPy. This project aims to explore the fundamental concepts of neural networks, including forward propagation, backpropagation, gradient computation, and optimization, without relying on high-level deep learning frameworks.

The project also includes an interactive **Streamlit web interface** that allows users to experiment with different datasets and neural network configurations.

## Features

- **Custom neural network architecture:** Build networks by combining dense layers and activation functions.
- **Forward propagation:** Compute predictions through the network.
- **Backpropagation:** Calculate gradients and propagate them through layers.
- **Activation functions:** ReLU, Sigmoid, Tanh, and Softmax.
- **Loss functions:** Mean Squared Error (MSE), Binary Cross-Entropy (BCE), and Categorical Cross-Entropy.
- **Optimization algorithms:** Stochastic Gradient Descent (SGD) and Momentum.
- **Mini-batch training:** Train the network using shuffled mini-batches.
- **Gradient checking:** Compare analytical gradients with numerical approximations to verify gradient calculations.
- **Automated tests:** Test forward-pass behavior and gradient calculations.
- **Interactive interface:** Explore datasets, configure training parameters, visualize loss curves, and evaluate classification performance using Streamlit.

## Supported Datasets

The Streamlit interface provides several ready-to-use datasets:

- **Iris:** Multiclass classification of iris flower species.
- **Wine:** Multiclass classification based on wine characteristics.
- **Breast Cancer Wisconsin:** Binary classification.
- **Moons:** A synthetic dataset for exploring nonlinear classification.

## Project Structure

```text
neural-net-numpy/
├── src/
│   ├── __init__.py
│   ├── activations.py
│   ├── layers.py
│   ├── losses.py
│   ├── network.py
│   ├── optimizer.py
│   └── main.py
├── tests/
│   ├── test_forward.py
│   └── test_gradients.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/neural-net-numpy.git
cd neural-net-numpy
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If necessary, install the interface and dataset dependencies separately:

```bash
python -m pip install streamlit scikit-learn matplotlib pandas
```

## Run the Streamlit Interface

From the project root directory, run:

```bash
python -m streamlit run app.py
```

The application will open in your browser. You can select a dataset, adjust the training parameters, train the neural network, and inspect its performance.

## Run the Tests

Run the tests from the project root:

```bash
python -m pytest tests/ -v
```

The tests cover selected forward-pass behavior and gradient calculations.

## Technologies

- Python
- NumPy
- Streamlit
- scikit-learn
- Matplotlib
- pandas
- pytest

## Learning Objectives

This project was developed to gain a deeper understanding of how neural networks work internally, including:

- Matrix operations in neural network layers.
- The chain rule and backpropagation.
- Gradient computation and numerical gradient checking.
- Activation and loss functions.
- Parameter updates during optimization.
- Training and evaluating classification models.

## Future Improvements

Potential future improvements include:

- Additional optimizers such as Adam.
- More activation and loss functions.
- Improved numerical stability.
- Support for saving and loading trained models.
- Additional datasets and network architectures.

## License

This project is available under the MIT License if an MIT license is added to the repository. Otherwise, no license has been specified yet.