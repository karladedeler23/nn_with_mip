from model_setup import *
from utils.write_to import write_variables_to_file
from matplotlib import pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2, l1
from datetime import datetime

########################################################

### PREPROCESSING 

# Function to load and preprocess regression data (example using UCI dataset)
def load_and_preprocess_data_regression(n, random_nb):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    column_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
        'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
    ]
    data = pd.read_csv(url, delim_whitespace=True, names=column_names)
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    # Nornalize input and ouput
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y).flatten()
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Select only n training points
    selected_indices = [i + random_nb for i in range(n)]
    X_train_sample = X_train[selected_indices]
    y_train_sample = y_train[selected_indices]

    ''' print("Shapes:")
    print(f"Training set: {X_train_sample.shape[0]}")
    print(f"Testing set: {X_test.shape[0]}") '''

    return X_train_sample, X_test, y_train_sample, y_test

def compute_big_M(input_dim, hidden_layers):
    M = [10*input_dim+1]
    for i in range(len(hidden_layers)):
        M.append(M[i]*hidden_layers[i]*10+1)
    return M

########################################################

### TRAINING NN WITH MIP

# Function to train Gurobi model for regression
def train_gurobi_model_regression(X_train, y_train, input_dim, hidden_layers, output_dim, loss_function, type_reg, lambda_reg, new = False):
    n = X_train.shape[0]
    M = compute_big_M(input_dim, hidden_layers)

    model = initialize_model()

    weights, biases, hidden_vars, relu_activation, binary_vars, y_pred = create_variables(model, input_dim, hidden_layers, output_dim, n)
    update_bound(y_pred, 'UB', 1)

    if hidden_layers != []:
        add_hidden_layer_constraints(model, X_train, weights, biases, hidden_vars, relu_activation, binary_vars, input_dim, hidden_layers, M, n)
        add_output_layer_constraints(model, relu_activation, weights, biases, hidden_vars, y_pred, binary_vars, output_dim, hidden_layers, M, n)

    else:
        add_output_layer_constraints(model, relu_activation, weights, biases, X_train, y_pred, binary_vars, output_dim, hidden_layers, M, n)
    
    if new == False: 
        set_loss_function_regression(model, weights, biases, loss_function, y_pred, y_train, n, output_dim, type_reg, lambda_reg)
    else:
        new_formulation(model, weights, biases, loss_function, n, y_pred, y_train, lambda_reg, type_reg)

    if optimize_model(model):
        W, b = extract_weights_biases(model, weights, biases)
        #write_variables_to_file(model, weights, biases, hidden_vars, binary_vars[-1], relu_activation, y_pred, hidden_layers, output_dim, n, 'src/utils/variables_values.txt')
        return model.Runtime, W, b
    else:
        return None, None, None

# Function to change the bound for a varibale within the model after creation, to maximise the reuse of previous function
def update_bound(var, bound, value):
    assert bound in ['LB', 'UB'], "Bound must be either 'LB' (Lower Bound) or 'UB' (Upper Bound)"
    for v in var.values():
        v.setAttr(bound, value)

# Function to calculate the error
def calculate_error(model, loss_function, n, y_pred, y_train_sample):
    loss_expr = 0
    
    if loss_function == 'mse':
        for i in range(n):
            loss_expr += 1/n * (y_pred[i, 0] - y_train_sample[i]) * (y_pred[i, 0] - y_train_sample[i])
    elif loss_function == 'mae':
        for i in range(n):
            diff_pred = model.addVars(n, vtype=GRB.CONTINUOUS, name=f"abs_diff_y_pred")
            model.addConstr(diff_pred[i] >= y_pred[i,0] - y_train_sample[i])
            model.addConstr(diff_pred[i] >= - (y_pred[i, 0] - y_train_sample[i]))
            loss_expr += 1/n * diff_pred[i]
    else:
        raise ValueError("Unsupported loss function")
    return loss_expr

# Function to regularisation L1 term, without the coefficient lambda
def calculate_reg_l1(model, weights, biases):
    print("REGULARISATION L1")
    abs_weights, abs_biases = [], []
    reg_expr = 0
    
    # Create absolute value variables for weights
    for i, weight_matrix in enumerate(weights):
        abs_weight_matrix = {}
        for j, k in weight_matrix.keys():
            abs_weight_matrix[j, k] = model.addVar(name=f'abs_weight_{i}_{j}_{k}', vtype=GRB.CONTINUOUS, lb=0, ub=1)
        abs_weights.append(abs_weight_matrix)
    
    # Create absolute value variables for biases
    for i, bias_matrix in enumerate(biases):
        abs_bias_matrix = {}
        for j in bias_matrix.keys():
            abs_bias_matrix[j] = model.addVar(name=f'abs_bias_{i}_{j}', vtype=GRB.CONTINUOUS, lb=0, ub=1)
        abs_biases.append(abs_bias_matrix)
    
    # Add the absolute values of the weights to the regularization term
    for i, weight_matrix in enumerate(weights):
        for (j, k) in weight_matrix.keys():
            model.addConstr(abs_weights[i][j, k] >= weight_matrix[j, k])
            model.addConstr(abs_weights[i][j, k] >= -weight_matrix[j, k])
            # Add regularization to the loss expression
            reg_expr += abs_weights[i][j, k]
    
    # Add the absolute values of the biases to the regularization term
    for i, bias_matrix in enumerate(biases):
        for j in bias_matrix.keys():
            model.addConstr(abs_biases[i][j] >= bias_matrix[j])
            model.addConstr(abs_biases[i][j] >= -bias_matrix[j])
            # Add regularization to the loss expression
            reg_expr += abs_biases[i][j]

    return reg_expr

# Function to regularisation L2 term, without the coefficient lambda
def calculate_reg_l2(weights, biases):
    print("REGULARISATION L2")
    reg_expr = 0

    # Add the absolute values of the weights to the regularization term
    for i, weight_matrix in enumerate(weights):
        for (j, k) in weight_matrix.keys():
            reg_expr += (weight_matrix[j, k] ** 2)
    
    # Add the absolute values of the biases to the regularization term
    for i, bias_matrix in enumerate(biases):
        for j in bias_matrix.keys():
            reg_expr +=  (bias_matrix[j] ** 2)
    return reg_expr

# Function to set loss function for regression
def set_loss_function_regression(model, weights, biases, loss_function, y_pred, y_train_sample, n, output_dim, reg, lambda_reg):
    # Adding loss function for regression
    loss_expr = gp.LinExpr()
    loss_expr = calculate_error(model, loss_function, n, y_pred, y_train_sample)

    # Adding regularization
    if lambda_reg != 0.0:
        if reg == 1 :
            loss_expr += lambda_reg * calculate_reg_l1(model, weights, biases)
        elif reg == 2 :
            loss_expr += lambda_reg * calculate_reg_l2(weights, biases)
    
    # Objective function
    model.setObjective(loss_expr, GRB.MINIMIZE)

# Function to create a new formulation
def new_formulation(model, weights, biases, error_function, n, y_pred, y_train_sample, threshold, reg):
    error = model.addVar(vtype=GRB.CONTINUOUS, name="error")
    error_def = gp.LinExpr()
    error_def += calculate_error(model, error_function, n, y_pred, y_train_sample)

    # Add a constraint to ensure the error is under the threshold
    model.addConstr(error == error_def, name="error_def")
    model.addConstr(error <= threshold, name="error_threshold")

    # Calculating the loss function = regularisation
    loss_expr = gp.LinExpr()
    loss_expr += 1
    if reg == 1 :
        loss_expr += calculate_reg_l1(model, weights, biases)
    elif reg == 2 :
        loss_expr += calculate_reg_l2(weights, biases)
    
    # Objective function
    model.setObjective(loss_expr, GRB.MINIMIZE)

########################################################

### CALCULATING THE ERROR

def compute_error_with_mip(X, y, weights, biases, loss_function):
    error = 0
    y_pred = [item[0] for item in predict_with_mip(weights, biases, X)]
    # print(f'{y_pred} instead of {y}')
    if loss_function == 'mse':
        # Compute Mean Squared Error
        error = np.mean((y_pred - y)**2)
    elif loss_function == 'mae':
        error = np.mean(np.abs(y_pred - y))
    else :
        raise ValueError("Unsupported loss function")
    return error

def compute_error_with_sgd(X, y, model, loss_function):
    error = 0
    y_pred = model.predict(X).flatten()
    # print(f'{y_pred} instead of {y}')
    if loss_function == 'mse':
        # Compute Mean Squared Error
        error = np.mean((y_pred - y)**2)
    elif loss_function == 'mae':
        error = np.mean(np.abs(y_pred - y))
    else :
        raise ValueError("Unsupported loss function")
    return error


########################################################

### RUNNING THE TRAINING

def run_regression_mip(X_train, y_train, hidden_layers, loss_function, type_reg, lambda_reg = 0.0, new = False):
    input_dim = X_train.shape[1]
    runtime, W, b = train_gurobi_model_regression(X_train, y_train, input_dim, hidden_layers, 1, loss_function, type_reg, lambda_reg, new)
    if W is not None and b is not None:
        print(f"Training completed in {runtime:.2f} seconds. \n")
    else:
        print("Training failed.")
    return W, b
    
def run_regression_sgd(X_train, y_train, hidden_layers, loss_function, type_reg, lambda_reg = 0.0):
    # Build the neural network model
    weight_constraint = ClipConstraint(min_value=-1, max_value=1)
    bias_constraint = ClipConstraint(min_value=-1, max_value=1)
    model_nn = Sequential()
    regularizer = l2(lambda_reg)
    if type_reg == 1:
        regularizer = l1(lambda_reg)
    # Input layer
    model_nn.add(Input(shape=(X_train.shape[1],)))
    model_nn.add(Dense(hidden_layers[0], activation='relu', kernel_regularizer=regularizer, kernel_constraint=weight_constraint, 
        bias_constraint=bias_constraint))
    # Hidden layers
    for units in hidden_layers[1:]:
        model_nn.add(Dense(units, activation='relu', kernel_regularizer=regularizer, kernel_constraint=weight_constraint, 
        bias_constraint=bias_constraint))
    # Output layer
    model_nn.add(Dense(1, activation='relu', kernel_regularizer=regularizer))
    # Compile the model with SGD optimizer
    sgd_optimizer = SGD(learning_rate=0.01)
    model_nn.compile(optimizer=sgd_optimizer, loss=loss_function)
    # Train the model
    model_nn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model_nn


########################################################

### PLOTTING FOR INSIGHTS IF NEEDED

def visualise_mip_vs_sgd(model_sgd, X, W, b, y):
    # Convert lists to NumPy arrays
    y_train = np.squeeze(y)  
    y_pred_train_sgd = np.squeeze(model_sgd.predict(X))
    y_pred_train_mip = np.squeeze(predict_with_mip(W, b, X))
    
    # Plotting the results
    margin = 0.1
    min_value = min(min(y_pred_train_sgd), min(y_pred_train_mip), min(y_train))
    max_value = max(max(y_pred_train_sgd), max(y_pred_train_mip), max(y_train))
    plt.figure(figsize=(8, 8))
    plt.plot(y_train, y_train, color='green', linestyle='--')
    plt.scatter(y_train, y_pred_train_sgd, color='blue', label='SGD Predictions', alpha=0.5)
    plt.scatter(y_train, y_pred_train_mip, color='red', label='MIP Predictions', alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Comparison of Model Predictions')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([min(y_train) - margin, max(y_train) + margin])
    plt.ylim([min_value - margin, max_value + margin])
    plt.legend()
    plt.show()
    
    # Plotting the residuals
    residual_sgd = y_train - y_pred_train_sgd
    residual_mip = y_train - y_pred_train_mip
    plt.figure(figsize=(12, 6))
    plt.scatter(y_train, residual_sgd, color='blue', label='SGD Residuals')
    plt.scatter(y_train, residual_mip, color='red', label='MIP Residuals')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Analysis')
    plt.legend()
    plt.show()

    return


########################################################

### MAIN FUNCTIONS

def main():
    random_nb = np.random.randint(390)
    print(random_nb)
    sample_size = 20 
    hidden_layers = [1]

    loss_function = 'mse'
    type_reg = 1
    lambda_reg = 0.1 

    X_train, X_test, y_train, y_test = load_and_preprocess_data_regression(sample_size, random_nb)
    W, b = run_regression_mip(X_train, y_train, hidden_layers, loss_function, type_reg, lambda_reg, True)
    model_sgd = run_regression_sgd(X_train, y_train, hidden_layers, loss_function, type_reg, lambda_reg)

    print("With MIP")
    error_train_mip = compute_error_with_mip(X_train, y_train, W, b, loss_function)
    error_test_mip = compute_error_with_mip(X_test, y_test, W, b, loss_function)
    print(f"Training {loss_function}: {error_train_mip:.2f}")
    print(f"Testing {loss_function}: {error_test_mip:.2f}\n")

    print("With SGD")
    error_train_sgd = compute_error_with_sgd(X_train, y_train, model_sgd, loss_function)
    error_test_sgd = compute_error_with_sgd(X_test, y_test, model_sgd, loss_function)
    print(f"Training {loss_function}: {error_train_sgd:.2f}")
    print(f"Testing {loss_function}: {error_test_sgd:.2f}\n")

    
    visualise_mip_vs_sgd(model_sgd, X_train, W, b, y_train)
    
    return

if __name__ == '__main__':
    main()