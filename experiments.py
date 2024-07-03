import numpy as np
import gurobipy as gp
from gurobipy import GRB
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from keras.constraints import Constraint
from tensorflow.keras.callbacks import Callback
from keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
import tensorflow as tf


########################################################

### PREPROCESSING 

# Function to load and preprocess data
def load_and_preprocess_data(n, random_nb):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    selected_indices = []
    
    for i in range(n):  # Iterate through the dataset to select one data point per class
        index = np.where(y_train == (i % 10))[0][random_nb + i]  # Get the index of one of the occurrences of the class
        selected_indices.append(index)
    X_train_sample = X_train[selected_indices]
    y_train_sample = y_train[selected_indices]

    # Flatten the inputs and normalize
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_train_sample = X_train_sample.reshape(X_train_sample.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    # Convert to "one-hot" vectors using the to_categorical function
    num_classes = 10
    y_train_one_hot = to_categorical(y_train, num_classes)
    y_train_sample_one_hot = to_categorical(y_train_sample, num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes)

    return (X_train_sample, y_train_sample, y_train_sample_one_hot), (X_test, y_test, y_test_one_hot), (X_train, y_train, y_train_one_hot)


########################################################

### TRAINING WITH SGD

# Define the custom max_correct loss function 
def custom_accuracy(y_true, y_pred):
    # Convert y_true to the correct format
    y_true_class = K.argmax(y_true, axis=-1)
    # Compute the predicted class
    y_pred_class = K.argmax(y_pred, axis=-1)
    # Compute the accuracy
    accuracy = K.sum(K.cast(K.equal(y_true_class, y_pred_class), dtype='float32'))
    return accuracy

# Define the custom hinge loss function
def hinge_loss(y_true, y_pred):
    hinge_loss = K.sum(K.maximum(0.0, 1 - (2 * y_true - 1) * y_pred))
    return hinge_loss

# To stop training when the loss is below a given variable:
class CustomStopper(Callback):
    def __init__(self, monitor='loss', value=6000, verbose=1):
        super(CustomStopper, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is not None and current < self.value:
            if self.verbose > 0:
                print(f"Epoch {epoch + 1}: early stopping threshold reached: {current:.4f}")
            self.model.stop_training = True

# To make sure the weights and biases are also between -1 and 1 like when using MIP
class ClipConstraint(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)
    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}

def train_sgd(X, y, y_one_hot, input_dim, hidden_layers, output_dim, loss_function):
    # Define the model with the constraints applied
    weight_constraint = ClipConstraint(min_value=-1, max_value=1)
    bias_constraint = ClipConstraint(min_value=-1, max_value=1)
    model_sgd = Sequential([
        Input(shape=(input_dim,)),
        Dense(hidden_layers[0], activation='relu', 
            kernel_constraint=weight_constraint, 
            bias_constraint=bias_constraint),
        Dense(output_dim, activation='relu', 
            kernel_constraint=weight_constraint, 
            bias_constraint=bias_constraint)
    ])

    # Define the loss function
    obj_function = 'categorical_crossentropy'
    if loss_function=='hinge':
        obj_function=hinge_loss

    # Define the custom callback to stop training when loss is below 10 times the variable
    #custom_stopper = CustomStopper(monitor='loss', value=0.9*X.shape[0]*10, verbose=1)

    model_sgd.compile(optimizer='adam', loss=obj_function, metrics=[custom_accuracy])
    model_sgd.fit(X, y_one_hot, epochs=10, batch_size=32, verbose=1) #callbacks=[custom_stopper])
    #accuracy_sgd = model_sgd.evaluate(X_test, y_test_one_hot, verbose=0)[1]
    #print("SGD Model Accuracy on testing set :", accuracy_sgd)

    # Extract weights and biases
    sgd_w, sgd_b = [], []
    for layer_idx, layer in enumerate(model_sgd.layers):
        weights, biases = layer.get_weights()
        sgd_w.append(weights)
        sgd_b.append(biases)

    # Forward pass to calculate the output
    hidden_layer_output = np.maximum(0, np.dot(X, sgd_w[0]) + sgd_b[0])  # ReLU activation
    #print(f"hidden_layer_output : {hidden_layer_output}")
    output_layer_output = np.maximum(0, np.dot(hidden_layer_output, sgd_w[1]) + sgd_b[1])  # ReLU activation
    print(f"output_layer_output : {output_layer_output}")
    print(f"Keras predictions loss: {model_sgd.predict(X)}")

    # Calculate hinge loss manually
    hinge_loss_per_class = np.maximum(1.0 - (2*y_one_hot-1) * output_layer_output, 0.0)
    hinge_loss_total = np.sum(hinge_loss_per_class)
    print(f"Manual hinge loss: {hinge_loss_total}")

    # Compare with Keras evaluation
    loss, accuracy = model_sgd.evaluate(X, y_one_hot, verbose=0)
    print(f"Keras evaluation loss: {loss}")
    
    return sgd_w, sgd_b


########################################################

### TRAINING NN WITH MIP

# Function to train Gurobi model with different loss functions
def train_gurobi_model(X_train_sample, y_train_sample, y_train_sample_one_hot, input_dim, hidden_layers, output_dim, M, margin, epsilon, loss_function):
    n = X_train_sample.shape[0]
    model = initialize_model()

    weights, biases, hidden_vars, relu_activation, binary_vars, y_pred = create_variables(model, input_dim, hidden_layers, output_dim, n)
    add_hidden_layer_constraints(model, X_train_sample, weights, biases, hidden_vars, relu_activation, binary_vars, input_dim, hidden_layers, M, n)
    add_output_layer_constraints(model, relu_activation, weights, biases, hidden_vars, y_pred, binary_vars, output_dim, hidden_layers, M, n)
    set_loss_function(model, y_pred, y_train_sample_one_hot, loss_function, M, margin, epsilon, n, output_dim)

    if optimize_model(model):
        return extract_weights_biases(model, weights, biases)
    else:
        return None, None

# Function to train Gurobi model with different loss functions and by startiing with some "good" weights and biases
def warm_start_train_gurobi_model(X_train_sample, y_train_sample, y_train_sample_one_hot, X_train, y_train, y_train_one_hot, input_dim, hidden_layers, output_dim, M, margin, epsilon, loss_function):
    n = X_train_sample.shape[0]
    model = initialize_model()

    weights, biases, hidden_vars, relu_activation, binary_vars, y_pred = create_variables(model, input_dim, hidden_layers, output_dim, n)

    W_init, b_init = train_sgd(X_train_sample, y_train_sample, y_train_sample_one_hot, input_dim, hidden_layers, output_dim, loss_function)
    # Initialise the weights and biases from results of SGD training
    for j in range(hidden_layers[0]):
        biases[0][j].start = b_init[0][j]
        for k in range(input_dim):
            weights[0][j, k].start = W_init[0][k,j]
    for l in range(1, len(hidden_layers)):
        for j in range(hidden_layers[l]):
            biases[l][j].start = b_init[l][j]
            for k in range(hidden_layers[l-1]):
                weights[l][j, k].start = W_init[l][k,j]
    for j in range(output_dim):
        biases[-1][j].start = b_init[-1][j]
        for k in range(hidden_layers[-1]):
            weights[-1][j, k].start = W_init[-1][k,j]

    add_hidden_layer_constraints(model, X_train_sample, weights, biases, hidden_vars, relu_activation, binary_vars, input_dim, hidden_layers, M, n)
    add_output_layer_constraints(model, relu_activation, weights, biases, hidden_vars, y_pred, binary_vars, output_dim, hidden_layers, M, n)
    set_loss_function(model, y_pred, y_train_sample_one_hot, loss_function, M, margin, epsilon, n, output_dim)

    if optimize_model(model):
        return extract_weights_biases(model, weights, biases)
    else:
        return None, None


########################################################

### AUXILIARY FUNCTIONS TO SET GUROBI MODEL

# Initialize model and set some parameters for the resolution
def initialize_model():
    model = gp.Model("neural_network_training")
    model.setParam('IntegralityFocus', 1)
    model.setParam('OptimalityTol', 1e-9)
    model.setParam('FeasibilityTol', 1e-9)
    model.setParam('MIPGap', 0)
    model.setParam('NodeLimit', 1e9)
    model.setParam('SolutionLimit', 1e9)
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
def add_hidden_layer_constraints(model, X_train_sample, weights, biases, hidden_vars, relu_activation, binary_vars, input_dim, hidden_layers, M, n):   
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
def add_output_layer_constraints(model, relu_activation, weights, biases, hidden_vars, y_pred, binary_vars, output_dim, hidden_layers, M, n):
    for i in range(n):
        for j in range(output_dim):
            model.addConstr(hidden_vars[-1][i, j] == gp.quicksum(relu_activation[-1][i, k] * weights[-1][j, k] for k in range(hidden_layers[-1])) + biases[-1][j])
            model.addConstr(y_pred[i, j] >= hidden_vars[-1][i, j])
            model.addConstr(y_pred[i, j] >= 0)
            model.addConstr(y_pred[i, j] <= hidden_vars[-1][i, j] + M * (1 - binary_vars[-1][i, j]))
            model.addConstr(y_pred[i, j] <= M * binary_vars[-1][i, j])

# Define the loss function based on the choice
def set_loss_function(model, y_pred, y_train_sample_one_hot, loss_function, M, margin, epsilon, n, output_dim):
    loss_expr = gp.LinExpr()
    if loss_function == 'max_correct':
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
            true_class = np.argmax(y_train_sample_one_hot[i])
            model.addConstr(correct_preds[i] == predicted_class[i, true_class], name=f"correct_pred_{i}")
            loss_expr -= correct_preds[i]
    elif loss_function == 'sat_margin':
        # Define binary variables to indicate correct predictions
        correct_preds = model.addVars(n, output_dim, vtype=GRB.BINARY, name="correct_preds")
        for i in range(n):
            for j in range(output_dim):
                y_true = 2 * y_train_sample_one_hot[i, j] - 1
                # If correct_preds[i, j] == 1, then y_true * y_pred[i, j] >= margin
                model.addConstr(y_true * y_pred[i, j] >= margin - M * (1 - correct_preds[i, j]))
                # If correct_preds[i, j] == 0, then y_true * y_pred[i, j] < margin
                model.addConstr(-y_true * y_pred[i, j] <= margin - epsilon + M * correct_preds[i, j])
                # Accumulate the binary variables for the loss expression
                loss_expr += 1 - correct_preds[i, j]
    elif loss_function == 'hinge':
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
                loss_expr += hinge_loss_terms[i, j]
    else:
        raise ValueError("Unsupported loss function")

    # Objective function
    model.setObjective(loss_expr, GRB.MINIMIZE)

def optimize_model(model):
    model.optimize()
    return model.status == GRB.OPTIMAL

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

### MAIN FUNCTION
# Function to run the entire process multiple times and calculate average accuracy
def run_multiple_experiments_warm_start(num_experiments, sample_size, hidden_layers, M, margin, epsilon, loss_function, random_nb, warm_start = False):
    training_accuracies = []
    testing_accuracies = []

    for _ in range(num_experiments):
        # Load and preprocess data
        (X_train_sample, y_train_sample, y_train_sample_one_hot), (X_test, y_test, y_test_one_hot), (X_train, y_train, y_train_one_hot) = load_and_preprocess_data(sample_size, random_nb)
        
        # Train Gurobi model and get optimal weights and biases
        if warm_start : 
            print('warm start')
            W_opt, b_opt = warm_start_train_gurobi_model(X_train_sample, y_train_sample, y_train_sample_one_hot, X_train, y_train, y_train_one_hot, X_train_sample.shape[1], hidden_layers, 10, M, margin, epsilon, loss_function)
        else :
            print('no warm start')
            W_opt, b_opt = train_gurobi_model(X_train_sample, y_train_sample, y_train_sample_one_hot, X_train_sample.shape[1], hidden_layers, 10, M, margin, epsilon, loss_function)

        if W_opt is not None and b_opt is not None:
            predictions_training = predict_with_mip(W_opt, b_opt, X_train_sample, y_train_sample_one_hot, y_train_sample)
            accuracy_training = accuracy_score(y_train_sample, predictions_training)
            training_accuracies.append(accuracy_training)
            predictions_testing = predict_with_mip(W_opt, b_opt, X_test, y_test_one_hot, y_test)
            accuracy_testing = accuracy_score(y_test, predictions_testing)
            testing_accuracies.append(accuracy_testing)
        else:
            print("Model did not converge.")

    return np.mean(training_accuracies), np.mean(testing_accuracies)