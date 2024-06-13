import numpy as np
import gurobipy as gp
from gurobipy import GRB
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

########################################################

# Function to load and preprocess data
def load_and_preprocess_data(n):
    (X_train_sample, y_train), (X_test, y_test) = mnist.load_data()
    selected_indices = []
    selected_labels = []
    for class_label in range(n):  # Iterate through the dataset to select one data point per class
        index = np.where(y_train == (class_label % 10))[0][np.random.randint(100)]  # Get the index of one of the occurrences of the class
        selected_indices.append(index)
        selected_labels.append(class_label)
    X_train_sample = X_train_sample[selected_indices]
    y_train_sample = y_train[selected_indices]

    # Flatten the inputs and normalize
    X_train_sample = X_train_sample.reshape(X_train_sample.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    # Convert to "one-hot" vectors using the to_categorical function
    num_classes = 10
    y_train_one_hot = to_categorical(y_train_sample, num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes)

    return (X_train_sample, y_train_sample, y_train_one_hot), (X_test, y_test, y_test_one_hot)

########################################################

# Function to train Gurobi model with different loss functions
def train_gurobi_model(X_train_sample, y_train_sample, y_train_one_hot, input_dim, hidden_layers, output_dim, M, margin, epsilon, loss_function):
    n = X_train_sample.shape[0]

    # Initialize model and set some parameters for the resolution
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
    y_pred = model.addVars(n, output_dim, vtype=GRB.CONTINUOUS, lb=0, name=f"y_pred")
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

    # Define the loss function based on the choice
    if loss_function == 'max_correct':
        # Max Correct  Loss Function
        loss_expr = gp.LinExpr()
        # Variables: Binary indicators for correct predictions
        correct_preds = model.addVars(n, vtype=GRB.BINARY, name="correct_preds")
        # Variables: Predicted class for each sample
        predicted_class = model.addVars(n, output_dim, vtype=GRB.BINARY, name="predicted_class")
        # Constraints to ensure that for each sample, exactly one class is predicted
        for i in range(n):
            model.addConstr(gp.quicksum(predicted_class[i, j] for j in range(output_dim)) == 1, name=f"unique_class_{i}")
        # Constraints to ensure that the predicted class has the highest score
        for i in range(n):
            for j in range(output_dim):
                for k in range(output_dim):
                    if j != k:
                        model.addConstr(y_pred[i, j] - y_pred[i, k] >= epsilon - M * (1 - predicted_class[i, j]), 
                                        name=f"max_class_{i}_{j}_{k}")
        # Constraints to ensure correct_preds is set correctly
        for i in range(n):
            true_class = np.argmax(y_train_one_hot[i])
            model.addConstr(correct_preds[i] == predicted_class[i, true_class], name=f"correct_pred_{i}")
            loss_expr -= correct_preds[i]
    elif loss_function == 'sat_margin':
        # SAT Margin Loss Function
        loss_expr = gp.LinExpr()
        # Define binary variables to indicate correct predictions
        correct_preds = model.addVars(n, output_dim, vtype=GRB.BINARY, name="correct_preds")
        for i in range(n):
            for j in range(output_dim):
                y_true = 2 * y_train_one_hot[i, j] - 1
                # If correct_preds[i, j] == 1, then y_true * y_pred[i, j] >= margin
                model.addConstr(y_true * y_pred[i, j] >= margin - M * (1 - correct_preds[i, j]))
                # If correct_preds[i, j] == 0, then y_true * y_pred[i, j] < margin
                model.addConstr(-y_true * y_pred[i, j] <= margin - epsilon + M * correct_preds[i, j])
                # Accumulate the binary variables for the loss expression
                loss_expr += 1 - correct_preds[i, j]
    elif loss_function == 'hinge':
        # Hinge Loss Function
        loss_expr = gp.LinExpr()
        # Define auxiliary variables for hinge loss terms
        hinge_loss_terms = model.addVars(n, output_dim, vtype=GRB.CONTINUOUS, name="hinge_loss_terms")
        # Constraints for hinge loss
        for i in range(n):
            for j in range(output_dim):
                # True class label (-1 or 1)
                y_true = 2 * y_train_one_hot[i, j] - 1
                # Hinge loss constraint
                model.addConstr(hinge_loss_terms[i, j] >= 0)
                model.addConstr(hinge_loss_terms[i, j] >= (1 - y_true * y_pred[i, j]))
                loss_expr += hinge_loss_terms[i, j]
    else:
        raise ValueError("Unsupported loss function")

    # Objective function
    model.setObjective(loss_expr, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()

    # Extract weights and biases if the model is optimal
    if model.status == GRB.OPTIMAL:
        W_opt, b_opt = extract_weights_biases(model, weights, biases)
        return W_opt, b_opt
    else:
        return None, None

########################################################

# Function to extract weights and biases
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

########################################################

# Predict the digits with the MIP model
def predict_with_mip(W_opt, b_opt, X, y, true_labels):
    predictions = []

    for i in range(X.shape[0]):
        sample = X[i]
        layer_output = sample
        #print(f"Initial input: {layer_output}")

        for l in range(len(W_opt)):
            W_l = W_opt[l]
            b_l = b_opt[l]

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

########################################################

# Function to run the entire process multiple times and calculate average accuracy
def run_multiple_experiments(num_experiments, sample_size, hidden_layers, M, margin, epsilon, loss_function):
    accuracies = []
    for _ in range(num_experiments):
        # Load and preprocess data
        (X_train_sample, y_train_sample, y_train_one_hot), (X_test, y_test, y_test_one_hot) = load_and_preprocess_data(sample_size)
        
        # Train Gurobi model and get optimal weights and biases
        W_opt, b_opt = train_gurobi_model(X_train_sample, y_train_sample, y_train_one_hot, X_train_sample.shape[1], hidden_layers, 10, M, margin, epsilon, loss_function)

        if W_opt is not None and b_opt is not None:
            # Create and evaluate Keras model
            predictions = predict_with_mip(W_opt, b_opt, X_train_sample, y_train_one_hot, y_train_sample)
            accuracy = accuracy_score(y_train_sample, predictions)
            accuracies.append(accuracy)
        else:
            print("Model did not converge.")

    average_accuracy = np.mean(accuracies)
    return average_accuracy
