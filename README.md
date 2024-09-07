# Training a neural network with mixed-integer optimisation

## Project Overview
This master reasearch project investigates the application of mixed-integer optimisation (MIP) to train neural networks by formulating the training phase as an optimisation problem. While MIP has been explored for network validation and verification, using it for neural network training is a novel approach with challenges, particularly in terms of scalability. This study explores the formulation and the impact of different loss functions, warm-start initialisation, and regularisation on solution quality, training efficiency, and convergence speed.

## Project Objectives
- To explore MIP for training neural networks with continuous weights and biases, providing more certainty in machine learning problem formulations.
- To try to address the scalability challenge of using MIP solvers (specifically, Gurobi) by investigating the impact of techniques like different loss functions, regularisation, and warm-start initialisation on training efficiency.

## Requirements
This project is built in Python and uses Gurobi as the MIP solver. Make sure Gurobi is installed, and you have an active license.

### Installing Dependencies
1. Install Gurobi from [Gurobi’s website](https://www.gurobi.com/), and obtain a license.
2. Clone the repository:
   ```bash
   git clone git@github.com:karladedeler23/nn_with_mip.git
3. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
4. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure
```bash
root/
│
├── README.md               # This file
├── requirements.txt        # Required Python libraries
├── .gitignore              # Files to be ignored by Git
├── src/                    # Source code folder
│   ├── model_setup.py      # Global model setup
│   ├── classification.py   # Classification task script
│   ├── regression.py       # Regression task script
│   ├── plots/              # Scripts for generating plots
│   │   ├── plot_mip_vs_sgd.py       # MIP vs SGD comparison
│   │   ├── plot_regularization.py   # Effect of regularization
│   │   ├── plot_warm_start.py       # Warm start technique
│   │   ├── plot_loss_landscape.py   # Loss landscape exploration
│   │   ├── plot_pen_vs_mnist.py     # Comparison on Pen and MNIST datasets
│   └── utils/              # Utility functions (optional)
├── graphs/                 # Folder to store generated plots
```

## Running the Code

### Training the Models
The main functions for the classification or regression tasks are in the corresponding ```.py``` script.  ```src/regression.py``` has a main function which shows an example of how to use thhem.  ```src/classification.py``` does not but examples are available in the ```src/plots``` folder.

### Generating Plots for Classification

The project contains scripts to generate various plots that analyse different aspects of the training. Running these scripts will produce visualisations stored in the `graphs/` folder. Use the following commands to generate each type of plot:

- **MIP vs SGD comparison**:
   ```bash
   python src/plots/plot_mip_vs_sgd.py
- **Comparison of different loss functions**:
   ```bash
   python src/plots/plot_diff_loss.py
- **Comparison on Pen and MNIST datasets**:
   ```bash
   python src/plots/plot_pen_vs_mnist.py
- **Warm start technique**:
    ```bash
    python src/plots/plot_warm_start.py
- **Loss landscape exploration**:
    ```bash
    python src/plots/plot_loss_landscape.py
- **Effect of L1 regularisation**:
    ```bash
    python src/plots/plot_regularisation.py
    ```


## License

This project is part of my Research Master thesis and is intended for academic use only.
For any inquiries or feedback, please contact me at [karla.dedeler23@ic.ac.uk](mailto:karla.dedeler23@ic.ac.uk).
