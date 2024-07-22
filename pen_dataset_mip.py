import numpy as np
import gurobipy as gp
from gurobipy import GRB
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from keras.constraints import Constraint
from keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd

########################################################

#### Definition of the parameters we might want to change 

n = 10  # number of data points
hidden_layers = [4]    # Definition of the neural network structure
M = [16, 16*hidden_layers[0]] # Big M constant for ReLU activation constraints (w and b are max = 1)
margin = M[-1]*0.01    # A reasonable margin (for SAT margin) should be a small fraction of this estimated output range
epsilon =  1.0e-4    # set the precision (for SAT margin)
lambda_reg = 1.0e-5  # Regularization parameter

random_nb = 709 #np.random.randint(1000)
print(random_nb)

########################################################

#### Loading and preprocessing the data

# Load the Pen-Based Recognition of Handwritten Digits dataset from UCI repository
url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra'
url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes'

# Read the CSV files
train_data = pd.read_csv(url_train, header=None)
test_data = pd.read_csv(url_test, header=None)

# Split into features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values
'''
# Inspect the shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
'''
# Select one data point per class for training sample
selected_indices = []
for i in range(n):  # Iterate through the dataset to select one data point per class
    class_indices = np.where(y_train == (i % 10))[0]
    if len(class_indices) > random_nb + i:
        index = class_indices[random_nb + i]  # Get the index of one of the occurrences of the class
    else:
        index = class_indices[0]  # In case the desired index is out of bounds, use the first index
    selected_indices.append(index)
X_train_sample = X_train[selected_indices]
y_train_sample = y_train[selected_indices]

# Normalize the inputs
X_train = X_train / 100
X_train_sample = X_train_sample / 100
X_test = X_test / 100
#print(X_train_sample[0])

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train_one_hot = to_categorical(y_train, num_classes)
y_train_sample_one_hot = to_categorical(y_train_sample, num_classes)
y_test_one_hot = to_categorical(y_test, num_classes)

# Definition of the neural network structure
input_dim = X_train_sample.shape[1] 
output_dim = num_classes


########################################################

#### Training using a Gurobi optimization model

# Initialise model and set some paramters for the resolution
model = gp.Model("neural_network_training")
#model.setParam('IntegralityFocus', 1)
#model.setParam('OptimalityTol', 1e-5)
#model.setParam('FeasibilityTol', 1e-9)
#model.setParam('MIPGap', 0)
#model.setParam('NodeLimit', 1e9)
#model.setParam('SolutionLimit', 1e9)

# Define variables for weights and biases
weights = []
biases = []

# Define weights and biases variables for each hidden layer
previous_layer_size = input_dim
for i, layer_size in enumerate(hidden_layers):
    W = model.addVars(layer_size, previous_layer_size, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=f"W{i+1}")
    b = model.addVars(layer_size, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=f"b{i+1}")
    weights.append(W)
    biases.append(b)
    previous_layer_size = layer_size

# Define variables for the hidden outputs
hidden_vars = []
relu_activation = []
binary_vars = []
for i, layer_size in enumerate(hidden_layers):
    z_hidden = model.addVars(n, layer_size, vtype=GRB.CONTINUOUS, lb= -GRB.INFINITY, name=f"z{i+1}")
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
z_hidden_final = model.addVars(n, output_dim, vtype=GRB.CONTINUOUS, lb= -GRB.INFINITY, name=f"z_final")
hidden_vars.append(z_hidden_final)
y_pred = model.addVars(n, output_dim, vtype=GRB.CONTINUOUS, lb=0, name=f"y_pred")
binary_v_output = model.addVars(n, output_dim, vtype=GRB.BINARY, name=f"binary_vars_final")
binary_vars.append(binary_v_output)

# Constraints for the first hidden layer
for i in range(n):
    for j in range(hidden_layers[0]):
        model.addConstr(hidden_vars[0][i, j] == gp.quicksum(X_train_sample[i, k] * weights[0][j, k] for k in range(input_dim)) + biases[0][j])
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
for i in range(n):
    for j in range(output_dim):
        model.addConstr(z_hidden_final[i, j] == gp.quicksum(relu_activation[-1][i, k] * W_output[j, k] for k in range(hidden_layers[-1])) + b_output[j])
        model.addConstr(y_pred[i, j] >= z_hidden_final[i, j])
        model.addConstr(y_pred[i, j] >= 0)
        model.addConstr(y_pred[i, j] <= z_hidden_final[i, j] + M[-1] * (1 - binary_v_output[i, j]))
        model.addConstr(y_pred[i, j] <= M[-1] * binary_v_output[i, j])

### LOSS FUNCTION
loss_expr = gp.LinExpr()

## HINGE LOSS FUNCTION
# Define auxiliary variables for hinge loss terms
hinge_loss_terms = model.addVars(n, output_dim, vtype=GRB.CONTINUOUS, name="hinge_loss_terms")

# Constraints for hinge loss
for i in range(n):
    for j in range(output_dim):
        # True class label (-1 or 1)
        y_true = 2 * y_train_sample_one_hot[i, j] - 1
        
        # Hinge loss constraint
        model.addConstr(hinge_loss_terms[i, j] >= 0)
        model.addConstr(hinge_loss_terms[i, j] >= (1 - y_true * y_pred[i, j]))

# Objective function
loss_expr = 1/n*gp.quicksum(hinge_loss_terms[i, j] for i in range(n) for j in range(output_dim))

'''
## SAT MARGIN LOSS FUNCTION

# Define binary variables to indicate correct predictions
correct_preds = model.addVars(n, output_dim, vtype=GRB.BINARY, name="correct_preds")

for i in range(n):
    for j in range(output_dim):
        y_true = 2 * y_train_sample_one_hot[i, j] - 1
        # If correct_preds[i, j] == 1, then y_true * y_pred[i, j] >= margin
        model.addConstr(y_true * y_pred[i, j] >= margin - M[-1] * (1 - correct_preds[i, j]))

        # If correct_preds[i, j] == 0, then y_true * y_pred[i, j] < margin
        model.addConstr(- y_true * y_pred[i, j] <= margin - epsilon + M[-1] * correct_preds[i, j])

        # Accumulate the binary variables for the loss expression
        loss_expr += 1 - correct_preds[i, j]
'''
'''
## REGULARISATION
print("REGULARISATION L1")
abs_weights, abs_biases = [], []

# Create absolute weight variables
previous_layer_size = input_dim
for i, layer_size in enumerate(hidden_layers + [output_dim]):
    abs_W = model.addVars(layer_size, previous_layer_size, vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"abs_W{i+1}")
    abs_b = model.addVars(layer_size, vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"abs_b{i+1}")
    abs_weights.append(abs_W)
    abs_biases.append(abs_b)
    previous_layer_size = layer_size

# Add the absolute values of the weights to the regularization term
for i, weight_matrix in enumerate(weights):
    for (j, k) in weight_matrix.keys():
        model.addConstr(abs_weights[i][j, k] >= weight_matrix[j, k])
        model.addConstr(abs_weights[i][j, k] >= -weight_matrix[j, k])
        # Add regularization to the loss expression
        loss_expr += lambda_reg * abs_weights[i][j, k]

# Add the absolute values of the biases to the regularization term
for i, bias_matrix in enumerate(biases):
    for (j) in bias_matrix.keys():
        model.addConstr(abs_biases[i][j] >= bias_matrix[j])
        model.addConstr(abs_biases[i][j] >= -bias_matrix[j])
        # Add regularization to the loss expression
        loss_expr += lambda_reg * abs_biases[i][j]
'''
# Objective function
model.setObjective(loss_expr, GRB.MINIMIZE)

# Save model for inspection
model.write('model.lp')

# Optimise the model
model.optimize()

########################################################

#### Write the results (for debugging mostly)

# Function to write variables to a file
def write_variables_to_file(model, filename):
    with open(filename, 'w') as f:
        # Write the values of weight variables
        for l in range(len(weights)):
            W = weights[l]
            #abs_W = abs_weights[l]
            for key in W.keys():
                f.write(f"Weight W{l+1}[{key}] = {W[key].X}\n")
                #f.write(f"abs_Weight abs_W{l+1}[{key}] = {abs_W[key].X}\n")

        # Write the values of bias variables
        for l in range(len(biases)):
            b = biases[l]
            #abs_b = abs_biases[l]
            for key in b.keys():
                f.write(f"Bias b{l+1}[{key}] = {b[key].X}\n")
                #f.write(f"abs_Bias abs_b{l+1}[{key}] = {abs_b[key].X}\n")

        # Write the values of auxiliary variables for prediction
        for i in range(n):
            for k in range(len(hidden_layers)):
                for j in range(hidden_layers[k]):
                    f.write(f"Auxiliary Variable for calculation of z = Wx + b [{i}, {j}] = {hidden_vars[k][i, j].X}\n")
                    f.write(f"Variable for hidden layer relu_activation[{i}, {j}] = {relu_activation[k][i, j].X}\n")

        # Write the values of auxiliary variables for prediction
        for i in range(n):
            for j in range(output_dim):
                f.write(f"Auxiliary Variable for calculation of z = Wx + b[{i}, {j}] = {hidden_vars[-1][i, j].X}\n")
                f.write(f"Binary_variable associated = {binary_v_output[i,j].X}\n")
                f.write(f"Prediction Variable y_pred[{i}, {j}] = {y_pred[i, j].X}\n")
        '''
        # Write the values of hinge loss terms
        for i in range(n):
            for j in range(output_dim):
                f.write(f"Hinge Loss Term hinge_loss_terms[{i}, {j}] = {hinge_loss_terms[i, j].X}\n")
        
        # Write the values of correct_pred variables
        for i in range(n):
            #for j in range(output_dim):
                f.write(f"correct prediction for sample {i} = {correct_preds[i].X}\n")
        
        for i in range(n):
            for j in range(output_dim):
                f.write(f"Predicted class sample {i} class {j} = {predicted_class[i, j].X}\n")
        '''

# Call the function to write variables to a file
write_variables_to_file(model, 'variables_values.txt')

#### Extract the results

# Ensure Gurobi variable extraction and reshaping
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

# Proceed to the extraction
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    W_opt, b_opt = extract_weights_biases(model, weights, biases)
    '''
    for l in range(len(W_opt)):
        print(f"Layer {l+1} weights non-zero count: {np.count_nonzero(W_opt[l])}")
    '''
else:
    print("No optimal solution found.")

########################################################

#### Test the training efficiency

# Predict the digits with the MIP model
def predict_with_mip(X, y, true_labels):
    predictions = []

    for i in range(X.shape[0]):
        sample = X[i]
        layer_output = sample
        #print(f"Initial input: {layer_output}")

        for l in range(len(W_opt)):
            W_l = W_opt[l]
            b_l = b_opt[l]
            '''
            print(f"W_{l} shape: {W_l.shape}")
            print(f"W_{l} values: \n{W_l}")
            print(f"b_{l} shape: {b_l.shape}")
            print(f"b_{l} values: \n{b_l}")
            print(f"Layer output before affine transformation: {layer_output}")
            '''
            # Affine transformation
            layer_output = np.dot(W_l, layer_output) + b_l
            #print(f"Layer output after affine transformation: {layer_output}")

            # ReLU activation
            layer_output = np.maximum(0.0, layer_output)
            #print(f"Layer output after ReLU activation: {layer_output}")

        #print(f"Prediction output : {layer_output}")
        pred = np.argmax(layer_output)
        predictions.append(pred)
        #print(f"Sample {i}: Prediction = {pred}, True Label = {true_labels[i]}")
    
    return predictions

# Calculate accuracy
predictions_training = predict_with_mip(X_train_sample, y_train_one_hot, y_train_sample)
accuracy_mip_training = accuracy_score(y_train_sample, predictions_training)
print("MIP Model Accuracy on training set:", accuracy_mip_training)
'''
print("Result inside the model")
for i in range(n):
    ## print([y_pred[i, j].X for j in range(output_dim)])
    predicted_class_index = np.argmax([y_pred[i, j].X for j in range(output_dim)])
    print(f"Sample {i}: Prediction = {predicted_class_index}, True Label = {y_train_sample[i]}")
'''
########################################################

#### Test the model obtained

# Calculate accuracy
predictions_test = predict_with_mip(X_test, y_test_one_hot, y_test)
accuracy_mip_test = accuracy_score(y_test, predictions_test)
print("MIP Model Accuracy on testing set:", accuracy_mip_test)