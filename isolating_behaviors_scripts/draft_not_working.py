import numpy as np
import gurobipy as gp
from gurobipy import GRB
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
import tensorflow as tf
from matplotlib import pyplot as plt

########################################################

#### Definition of the parameters we might want to change 

n = 10  # number of data points
hidden_layers = [32]    # Definition of the neural network structure
M = 2.6e4   # Big M constant for ReLU activation constraints (output range)
margin = 300    # A reasonable margin (for SAT margin) should be a small fraction of this estimated output range
epsilon = 1.0e-6    # set the precision

########################################################

#### Loading and preprocessing the data
 
# Load MNIST data
(X_train_sample, y_train), (X_test, y_test) = mnist.load_data()

# Select n points from the dataset
selected_indices = []
selected_labels = []
for class_label in range(n): # Iterate through the dataset to select one data point per class
    index = np.where(y_train == (class_label%10))[0][np.random.randint(100)]  # Get the index of one of the occurrences of the class
    selected_indices.append(index)
    selected_labels.append(class_label)
X_train_sample = X_train_sample[selected_indices]
y_train_sample = y_train[selected_indices]

# Flatten the inputs and normalise 
X_train_sample = X_train_sample.reshape(X_train_sample.shape[0], -1)/255.0
X_test = X_test.reshape(X_test.shape[0], -1)/255.0

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train_one_hot = keras.utils.to_categorical(y_train_sample, num_classes)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)

# Definition of the neural network structure
input_dim = X_train_sample.shape[1] 
output_dim = num_classes

'''
# Plot the selected samples
for i in range(n):
    plt.subplot(1, n, i+1)
    plt.imshow(X_train_sample[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()

# Print the shapes to verify
print("X_train_sample shape:", X_train_sample.shape)
print(X_train_sample[0])
print("y_train_one_hot shape:", y_train_one_hot.shape)
'''

########################################################

#### Training using a Gurobi optimization model

# Initialise model and set some paramters for the resolution
model = gp.Model("neural_network_training")
model.setParam('IntegralityFocus', 1)
model.setParam('OptimalityTol', 1e-9)
model.setParam('FeasibilityTol', 1e-9)
model.setParam('MIPGap', 0)
model.setParam('NodeLimit', 1e9)
model.setParam('SolutionLimit', 1e9)

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
    z_hidden = model.addVars(n, layer_size, vtype=GRB.CONTINUOUS, name=f"z{i+1}")
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
z_hidden_final = model.addVars(n, output_dim, vtype=GRB.CONTINUOUS, name=f"z_final")
hidden_vars.append(z_hidden_final)
y_pred = model.addVars(n, output_dim, vtype=GRB.CONTINUOUS, lb=0, ub=80, name=f"y_pred")
binary_v_output = model.addVars(n, output_dim, vtype=GRB.BINARY, name=f"binary_vars_final")
binary_vars.append(binary_v_output)

# Constraints for the first hidden layer
for i in range(n):
    for j in range(hidden_layers[0]):
        model.addConstr(hidden_vars[0][i, j] == gp.quicksum(X_train_sample[i, k] * weights[0][j, k] for k in range(input_dim)) + biases[0][j])
        model.addConstr(relu_activation[0][i, j] >= hidden_vars[0][i, j])
        model.addConstr(relu_activation[0][i, j] >= 0)
        model.addConstr(relu_activation[0][i, j] <= hidden_vars[0][i, j] + M * (1 - binary_vars[0][i, j]))
        model.addConstr(relu_activation[0][i, j] <= M * binary_vars[0][i, j])

# Constraints for subsequent hidden layers
for l in range(1, len(hidden_layers)):
    for i in range(n):
        for j in range(hidden_layers[l]):
            model.addConstr(hidden_vars[l][i, j] == gp.quicksum(relu_activation[l-1][i, k] * weights[l][j, k] for k in range(hidden_layers[l-1])) + biases[l][j])
            model.addConstr(relu_activation[l][i, j] >= hidden_vars[l][i, j])
            model.addConstr(relu_activation[l][i, j] >= 0)
            model.addConstr(relu_activation[l][i, j] <= hidden_vars[l][i, j] + M * (1 - binary_vars[l][i, j]))
            model.addConstr(relu_activation[l][i, j] <= M * binary_vars[l][i, j])

# Constraints for the output layer
for i in range(n):
    for j in range(output_dim):
        model.addConstr(z_hidden_final[i, j] == gp.quicksum(relu_activation[-1][i, k] * W_output[j, k] for k in range(hidden_layers[-1])) + b_output[j])
        model.addConstr(y_pred[i, j] >= z_hidden_final[i, j])
        model.addConstr(y_pred[i, j] >= 0)
        model.addConstr(y_pred[i, j] <= z_hidden_final[i, j] + M * (1 - binary_v_output[i, j]))
        model.addConstr(y_pred[i, j] <= M * binary_v_output[i, j])

''' Constraints to make sure the maximum in y_pred is unique, but it does not work '''
# Add constraint to ensure only one maximum value in y_pred for each sample
max_prob = model.addVars(n, vtype=GRB.CONTINUOUS, name="max_prob")  # Create variables to represent the maximum probability for each sample
is_max = model.addVars(n, output_dim, vtype=GRB.BINARY, name="is_max")  # Binary variables to indicate if y_pred[i, j] is equal to max_prob[i]
for i in range(n):
    model.addConstr(max_prob[i] == gp.max_([y_pred[i, k] for k in range(output_dim)]))  # Define max_prob[i] as the maximum probability for sample i
    for j in range(output_dim):
        model.addConstr(y_pred[i, j] - max_prob[i] <= 10 * (1 - is_max[i, j]))  # Set is_max to 1 if y_pred equals max_prob
        model.addConstr(y_pred[i, j] - max_prob[i] >= -10 * is_max[i, j])  # Set is_max to 0 otherwise
    model.addConstr(gp.quicksum(is_max[i, j] for j in range(output_dim)) == 1)  # Ensure only one maximum value

''' Cross entropy loss function is not adapted to this model 
    since final activation function is ReLU and 
    not Softmax or Sigmoid so y_pred not strictly between [0,1]'''
## CROSS ENTROPY LOSS FUNCTION

# Loss calculation using a piecewise-linear approximation of the logarithm
log_breaking_points = np.linspace(1.0e-4, 10, 200)
log_values = np.log(log_breaking_points)

loss = model.addVar(name="loss")
log_loss_terms = []

for i in range(n):
    for j in range(output_dim):
        log_term = model.addVar(lb=-GRB.INFINITY, name=f"log_term_{i}_{j}")
        model.addGenConstrPWL(y_pred[i, j], log_term, log_breaking_points, log_values, name="pwl_log")
        log_loss_terms.append((log_term, y_train_one_hot[i, j]))

# Loss function
model.addConstr(loss == -gp.quicksum(log_term * cond for log_term, cond in log_loss_terms), name="loss_function")

# Objective function
model.setObjective(loss, GRB.MINIMIZE)

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
            for key in W.keys():
                f.write(f"Weight W{l+1}[{key}] = {W[key].X}\n")

        # Write the values of bias variables
        for l in range(len(biases)):
            b = biases[l]
            for key in b.keys():
                f.write(f"Bias b{l+1}[{key}] = {b[key].X}\n")

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
                f.write(f"Prediction Variable y_pred[{i}, {j}] = {y_pred[i, j].X}\n")
        

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

        # print(f"Prediction output : {layer_output}")
        pred = np.argmax(layer_output)
        predictions.append(pred)
        # print(f"Sample {i}: Prediction = {pred}, True Label = {true_labels[i]}")
    
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

########################################################

### Comparison with typical SGD

# Model with Keras (SGD)
model_sgd = Sequential([
    Input(shape=(input_dim,)),
    Dense(hidden_layers[0], activation='relu'),
    Dense(output_dim, activation='relu')
])

model_sgd.compile(optimizer='adam', loss=tf.keras.losses.Hinge(), metrics=['accuracy'])
model_sgd.fit(X_train_sample, y_train_one_hot, epochs=10, batch_size=32, verbose=0)
accuracy_sgd = model_sgd.evaluate(X_test, y_test_one_hot, verbose=0)[1]
print("SGD Model Accuracy:", accuracy_sgd)
