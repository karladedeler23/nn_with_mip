import numpy as np
import gurobipy as gp
from gurobipy import GRB
from keras.constraints import Constraint
import tensorflow as tf
import os

########################################################

### AUXILIARY FUNCTIONS TO SET GUROBI MODEL

# Initialize model and set some parameters for the resolution
def initialize_model():
    model = gp.Model("neural_network_training")
    #model.setParam('MIPGap', 0)
    #model.setParam('IntegralityFocus', 1)
    #model.setParam('OptimalityTol', 1e-9)
    #model.setParam('FeasibilityTol', 1e-9)
    #model.setParam('NodeLimit', 1e9)
    #model.setParam('SolutionLimit', 1e9)
    return model

# Define variables to mimic the NN
def create_variables(model, input_dim, hidden_layers, output_dim, n):
    weights, biases, hidden_vars, relu_activation, binary_vars = [], [], [], [], []

    previous_layer_size = input_dim
    for i, layer_size in enumerate(hidden_layers):
        W = model.addVars(layer_size, previous_layer_size, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=f"W{i+1}")
        b = model.addVars(layer_size, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=f"b{i+1}")
        weights.append(W)
        biases.append(b)
        previous_layer_size = layer_size

    # Define variables for the hidden outputs
    for i, layer_size in enumerate(hidden_layers):
        z_hidden = model.addVars(n, layer_size, vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY, name=f"z{i+1}")
        hidden_vars.append(z_hidden)
        a = model.addVars(n, layer_size, vtype=GRB.CONTINUOUS, name=f"a{i+1}=max(0,z)")
        relu_activation.append(a)
        binary_v = model.addVars(n, layer_size, vtype=GRB.BINARY, name=f"binary_vars{i+1}")
        binary_vars.append(binary_v)

    # Define output layer variables
    W_output = model.addVars(output_dim, hidden_layers[-1], vtype=GRB.CONTINUOUS, lb=-1, ub=1, name="W_output")
    b_output = model.addVars(output_dim, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name="b_output")
    weights.append(W_output)
    biases.append(b_output)

    # Define the output layer variables for the final activation function (here ReLU)
    z_hidden_final = model.addVars(n, output_dim, vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY, name=f"z_final")
    hidden_vars.append(z_hidden_final)
    y_pred = model.addVars(n, output_dim, vtype=GRB.CONTINUOUS, lb=0, name=f"y_pred")
    binary_v_output = model.addVars(n, output_dim, vtype=GRB.BINARY, name=f"binary_vars_final")
    binary_vars.append(binary_v_output)

    return weights, biases, hidden_vars, relu_activation, binary_vars, y_pred

# Constraints for the  hidden layers
def add_hidden_layer_constraints(model, X_train, weights, biases, hidden_vars, relu_activation, binary_vars, input_dim, hidden_layers, M, n):   
    # Constraints for the first hidden layer
    for i in range(n):
        for j in range(hidden_layers[0]):
            model.addConstr(hidden_vars[0][i, j] == gp.quicksum(X_train[i, k] * weights[0][j, k] for k in range(input_dim)) + biases[0][j])
            model.addConstr(relu_activation[0][i, j] >= hidden_vars[0][i, j])
            model.addConstr(relu_activation[0][i, j] >= 0)
            model.addConstr(relu_activation[0][i, j] <= hidden_vars[0][i, j] + M[0] * (1 - binary_vars[0][i, j]))
            model.addConstr(relu_activation[0][i, j] <= M[0] * binary_vars[0][i, j])

    # Constraints for subsequent hidden layers
    for l in range(1, len(hidden_layers)):
        for i in range(n):
            for j in range(hidden_layers[l]):
                model.addConstr(hidden_vars[l][i, j] == gp.quicksum(relu_activation[l-1][i, k] * weights[l][j, k] for k in range(hidden_layers[l-1])) + biases[l][j])
                model.addConstr(relu_activation[l][i, j] >= hidden_vars[l][i, j])
                model.addConstr(relu_activation[l][i, j] >= 0)
                model.addConstr(relu_activation[l][i, j] <= hidden_vars[l][i, j] + M[l] * (1 - binary_vars[l][i, j]))
                model.addConstr(relu_activation[l][i, j] <= M[l] * binary_vars[l][i, j])

# Constraints for the output layer
def add_output_layer_constraints(model, relu_activation, weights, biases, hidden_vars, y_pred, binary_vars, output_dim, hidden_layers, M, n):
    for i in range(n):
        for j in range(output_dim):
            model.addConstr(hidden_vars[-1][i, j] == gp.quicksum(relu_activation[-1][i, k] * weights[-1][j, k] for k in range(hidden_layers[-1])) + biases[-1][j])
            model.addConstr(y_pred[i, j] >= hidden_vars[-1][i, j])
            model.addConstr(y_pred[i, j] >= 0)
            model.addConstr(y_pred[i, j] <= hidden_vars[-1][i, j] + M[-1] * (1 - binary_vars[-1][i, j]))
            model.addConstr(y_pred[i, j] <= M[-1] * binary_vars[-1][i, j])

# Function to optimize the MIP model
def optimize_model(model):
    model.optimize()
    return model.status == GRB.OPTIMAL


########################################################

### BUILDING A SGD MODEL

# To make sure the weights and biases are also between -1 and 1 like when using MIP
class ClipConstraint(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)
    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}


########################################################

### AUXILIARY FUNCTIONS TO DEAL WITH THE RESULTS OF THE MIP MODEL

# Adapt model's weights / biases variables to the right shape
def extract_weights_biases(model, weights, biases):
    extracted_weights = []
    extracted_biases = []

    for l in range(len(weights)):
        W = weights[l]
        weight_shape = (max([k[0] for k in W.keys()]) + 1, max([k[1] for k in W.keys()]) + 1)
        W_vals = np.zeros(weight_shape)  # Initialize with the correct shape
        for key in W.keys():
            W_vals[key] = W[key].X
        extracted_weights.append(W_vals)

        b = biases[l]
        bias_shape = (max(b.keys()) + 1,)
        b_vals = np.zeros(bias_shape)
        for key in b.keys():
            b_vals[key] = b[key].X
        extracted_biases.append(b_vals)

    return extracted_weights, extracted_biases

# Predict with the MIP model
def predict_with_mip(W_opt, b_opt, X):
    predictions = []
    for i in range(X.shape[0]):
        sample = X[i]
        layer_output = sample
        for l in range(len(W_opt)):
            W_l = W_opt[l]
            b_l = b_opt[l]
            # Affine transformation
            layer_output = np.dot(W_l, layer_output) + b_l
            # ReLU activation
            layer_output = np.maximum(0.0, layer_output)
        predictions.append(layer_output)
    return predictions