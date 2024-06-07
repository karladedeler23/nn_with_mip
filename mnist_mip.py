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
from matplotlib import pyplot as plt

#### Loading and preprocessing the data
 
# Load MNIST data
(X_train_sample, y_train), (X_test, y_test) = mnist.load_data()

# Select n points from the dataset
n = 10 # number of data points
selected_indices = []
selected_labels = []
for class_label in range(n): # Iterate through the dataset to select one data point per class
    index = np.where(y_train == class_label)[0][np.random.randint(100)]  # Get the index of one of the occurrence of the class
    selected_indices.append(index)
    selected_labels.append(class_label)
X_train_sample = X_train_sample[selected_indices]
y_train_sample = y_train[selected_indices]

# Flatten the inputs and normalise 
X_train_sample = X_train_sample.reshape(X_train_sample.shape[0], -1)/255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1)/255.0

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train_one_hot = keras.utils.to_categorical(y_train_sample, num_classes)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)

# Definition of the neural network structure
input_dim = X_train_sample.shape[1] 
hidden_layers = [32] 
output_dim = 10  # MNIST has 10 classes

'''
# Plot the selected samples
for i in range(n):
    plt.subplot(1, n, i+1)
    plt.imshow(X_sample[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()

# Print the shapes to verify
print("X_train_sample shape:", X_train_sample.shape)
print("y_train_one_hot shape:", y_train_one_hot.shape)
'''

########################################################

#### Training using a Gurobi optimization model

# Initialize model
model = gp.Model("neural_network_training")

# Set some parameters / strategies to ensure exact optimal solution
model.setParam('MIPFocus', 3)       # Balance between finding new solutions and improving bounds
model.setParam('Heuristics', 0)     # Turn off heuristics
model.setParam('Cuts', 0)           # Turn off cuts
model.setParam('Presolve', 0)       # Turn off presolve
model.setParam('MIPGap', 0)         # Ensure no gap is allowed
model.setParam('MIPGapAbs', 0)      # Ensure no absolute gap is allowed

# Define variables for weights and biases
weights = []
biases = []

# Define variables for each layer
previous_layer_size = input_dim
for i, layer_size in enumerate(hidden_layers):
    W = model.addVars(layer_size, previous_layer_size, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=f"W{i+1}")
    b = model.addVars(layer_size, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name=f"b{i+1}")
    weights.append(W)
    biases.append(b)
    previous_layer_size = layer_size

# Define output layer variables
W_output = model.addVars(output_dim, hidden_layers[-1], vtype=GRB.CONTINUOUS, lb=-1, ub=1,name="W_output")
b_output = model.addVars(output_dim, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name="b_output")
weights.append(W_output)
biases.append(b_output)

# Define variables for the hidden layer outputs
hidden_outputs = []
for i, layer_size in enumerate(hidden_layers):
    h = model.addVars(n, layer_size, vtype=GRB.CONTINUOUS, lb=0, name=f"h{i+1}")
    hidden_outputs.append(h)

# Define the output layer output
y_pred = model.addVars(n, output_dim, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y_pred")

# Constraints for the forward pass
for i in range(n):  # copy of the neural network!
    # Input to first hidden layer
    for j in range(hidden_layers[0]):
        linear_expr = gp.LinExpr(gp.quicksum(weights[0][j, k] * X_train_sample[i, k] for k in range(input_dim)) + biases[0][j])
        aux_var = model.addVar(vtype=GRB.CONTINUOUS, name=f"aux_hidden0_{i}_{j}")
        model.addConstr(aux_var == linear_expr, name=f"aux_constr_hidden0_{i}_{j}")
        model.addGenConstrMax(hidden_outputs[0][i, j], [aux_var, 0], name=f"ReLU_hidden0_{i}_{j}") # ReLU function

    # Hidden layers
    for l in range(1, len(hidden_layers)):
        for j in range(hidden_layers[l]):
            linear_expr = gp.LinExpr(gp.quicksum(weights[l][j, k] * hidden_outputs[l-1][i, k] for k in range(hidden_layers[l-1])) + biases[l][j])
            aux_var = model.addVar(vtype=GRB.CONTINUOUS, name=f"aux_hidden{l}_{i}_{j}")
            model.addConstr(aux_var == linear_expr, name=f"aux_constr_hidden{l}_{i}_{j}")
            model.addGenConstrMax(hidden_outputs[l][i, j], [aux_var, 0], name=f"ReLU_hidden{l}_{i}_{j}") # ReLU function

    # Output layer
    for j in range(output_dim):
        model.addConstr(y_pred[i, j] == gp.quicksum(weights[-1][j, k] * hidden_outputs[-1][i, k] for k in range(hidden_layers[-1])) + biases[-1][j])

'''
# Constraints for non-zero weights
min_magnitude = 1.0e-03
for W in weights:
    for key in W.keys():
        model.addConstr(W[key]**2>= min_magnitude, name = f"nonzero{W[key]}")
'''

# Loss calculation using a piecewise-linear approximation of the logarithm
loss = model.addVar(name="loss")
log_loss_terms = []

for i in range(n):
    for j in range(output_dim):
        log_term = model.addVar(lb=-GRB.INFINITY, name=f"log_term_{i}_{j}")
        model.addGenConstrLog(y_pred[i, j], log_term, name=f"log_constr_{i}_{j}")
        log_loss_terms.append((log_term, y_train_one_hot[i, j]))

# Loss function
model.addConstr(loss == -gp.quicksum(log_term * cond for log_term, cond in log_loss_terms), name="loss_function")

# Objective function
model.setObjective(loss, GRB.MINIMIZE)

# Save model for inspection
#model.write('model.lp')

# Optimise the model
model.optimize()

########################################################

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
if model.status == GRB.INFEASIBLE:
    print("Model is infeasible")
    model.computeIIS()
    model.write("model.ilp")
elif model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    W_opt, b_opt = extract_weights_biases(model, weights, biases)
    
    for l in range(len(W_opt)):
        print(f"Layer {l+1} weights non-zero count: {np.count_nonzero(W_opt[l])}")
    
else:
    print("No optimal solution found.")

########################################################

#### Test the training efficiency

# Predict the digits with the MIP model
def predict_with_mip(X, y, true_labels):
    predictions = []

    # Make predictions for each test sample
    for i in range(X.shape[0]):
        sample = X[i]
        
        # Forward pass through the neural network
        layer_output = sample
        for l in range(len(hidden_layers)):
            W_l = W_opt[l]
            b_l = b_opt[l]
            
            layer_output = np.maximum(0, np.dot(W_l, layer_output) + b_l)
        
        W_out = W_opt[-1]
        b_out = b_opt[-1]
        
        pred = np.argmax(np.dot(W_out, layer_output) + b_out)
        #print(np.dot(W_out, layer_output) + b_out)
        predictions.append(pred)
        #print(f"Sample {i}: Prediction = {pred}, True Label = {true_labels[i]}")
    return predictions

# Calculate accuracy
predictions_training = predict_with_mip(X_train_sample, y_train_one_hot, y_train_sample)
accuracy_mip_training = accuracy_score(y_train_sample, predictions_training)
print("MIP Model Accuracy on training set:", accuracy_mip_training)

########################################################

#### Test the model obtained

# Calculate accuracy
predictions_test = predict_with_mip(X_test_flat, y_test_one_hot, y_test)
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

model_sgd.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_sgd.fit(X_train_sample, y_train_one_hot, epochs=10, batch_size=32, verbose=0)
accuracy_sgd = model_sgd.evaluate(X_test_flat, y_test_one_hot, verbose=0)[1]
print("SGD Model Accuracy:", accuracy_sgd)