from experiments import *
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
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
    y = data.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select only n training points
    selected_indices = [i + random_nb for i in range(n)]
    X_train_sample = X_train[selected_indices]
    y_train_sample = y_train[selected_indices]
    '''
    print("Shapes:")
    print(f"Training set: {X_train_sample.shape[0]}")
    print(f"Testing set: {X_test.shape[0]}")
   '''
    return X_train_sample, X_test, y_train_sample, y_test


def compute_big_M(input_dim, hidden_layers):
    M = [input_dim+1]
    for i in range(len(hidden_layers)):
        M.append(M[i]*hidden_layers[i]+1)
    return M

########################################################

### TRAINING NN WITH MIP

# Function to train Gurobi model for regression
def train_gurobi_model_regression(X_train_sample, y_train_sample, input_dim, hidden_layers, output_dim, M, lambda_reg):
    n = X_train_sample.shape[0]
    model = initialize_model()

    weights, biases, hidden_vars, relu_activation, binary_vars, y_pred = create_variables(model, input_dim, hidden_layers, output_dim, n)
    add_hidden_layer_constraints(model, X_train_sample, weights, biases, hidden_vars, relu_activation, binary_vars, input_dim, hidden_layers, M, n)
    add_output_layer_constraints_regression(model, relu_activation, weights, biases, hidden_vars, y_pred, binary_vars, output_dim, hidden_layers, M, n)
    set_loss_function_regression(model, weights, biases, y_pred, y_train_sample, M, n, input_dim, hidden_layers, output_dim, lambda_reg)

    if optimize_model(model):
        W, b = extract_weights_biases(model, weights, biases)
        return model.Runtime, W, b
    else:
        return None, None, None

# Function to add constraints for the output layer for regression
def add_output_layer_constraints_regression(model, relu_activation, weights, biases, hidden_vars, y_pred, binary_vars, output_dim, hidden_layers, M, n):
    # Adding constraints for the output layer (regression)
    for i in range(n):
        for j in range(output_dim):
            model.addConstr(
                y_pred[i, j] == gp.quicksum(weights[-1][j, k] * relu_activation[-1][i, k] for k in range(hidden_layers[-1])) + biases[-1][j],
                name=f"output_layer_{i}_{j}"
            )

# Function to set loss function for regression
def set_loss_function_regression(model, weights, biases, y_pred, y_train_sample, M, n, input_dim, hidden_layers, output_dim, lambda_reg):
    # Adding loss function for regression (MSE)
    loss_expr = gp.LinExpr()
    for i in range(n):
        loss_expr += 1/n * (y_pred[i, 0] - y_train_sample[i]) * (y_pred[i, 0] - y_train_sample[i])
    
    # Adding L2 regularization
    if lambda_reg != 0.0:
        print("REGULARISATION L2")
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
                loss_expr += lambda_reg * abs_weights[i][j, k] ** 2
        
        # Add the absolute values of the biases to the regularization term
        for i, bias_matrix in enumerate(biases):
            for j in bias_matrix.keys():
                model.addConstr(abs_biases[i][j] >= bias_matrix[j])
                model.addConstr(abs_biases[i][j] >= -bias_matrix[j])
                # Add regularization to the loss expression
                loss_expr += lambda_reg * abs_biases[i][j] ** 2
    
    # Objective function
    model.setObjective(loss_expr, GRB.MINIMIZE)

########################################################

# Function to compute the accuracy (Mean Squared Error)
def compute_accuracy(X, y, weights, biases):
    # Predict on the test set
    y_pred = [item[0] for item in predict_with_mip(weights, biases, X, y)]
    
    # Compute Mean Squared Error
    mse = np.mean((y_pred - y)**2)
    return mse

########################################################

### MAIN FUNCTION TO RUN THE TRAINING

def run_regression_mip(current_date_time, num_experiments, sample_size, hidden_layers, random_nb, lambda_reg = 0.0, warm_start = False, W_init = None, b_init = None):
    training_accuracies, testing_accuracies = [], []
    runtimes = []
    nn_config = {'hidden layers': hidden_layers,
                'training set size' : sample_size,
                'starting point in the data': random_nb,
                'loss' : 'mean squared error',                
                'Regularisation' : lambda_reg,
                'Warm start' : warm_start
                }

    for i in range(num_experiments):
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data_regression(sample_size, random_nb+i*sample_size)
        input_dim = X_train.shape[1]
        M = compute_big_M(input_dim, hidden_layers)

        # Train Gurobi model and get optimal weights and biases
        if warm_start and W_init is not None and b_init is not None : 
            print('warm start')
            # Using previous weights (when we are actually training with more points)
            # runtime, W, b = train_gurobi_model_regression_warm_start(X_train, y_train, input_dim, hidden_layers, 1, M, lambda_reg, W_init, b_init)
        else :
            print('no warm start')
            runtime, W, b = train_gurobi_model_regression(X_train, y_train, input_dim, hidden_layers, 1, M, lambda_reg)
        if W is not None and b is not None:
            runtimes.append(runtime)
            print(f"Training completed in {runtime:.2f} seconds. \n")
            # Compute and print the test accuracy
            train_mse = compute_accuracy(X_train, y_train, W, b)
            training_accuracies.append(train_mse)
            test_mse = compute_accuracy(X_test, y_test, W, b)
            testing_accuracies.append(test_mse)
            print("MIP")
            print(f"Train Mean Squared Error: {train_mse:.2f}")
            print(f"Test Mean Squared Error: {test_mse:.2f} \n")
        else:
            print("Training failed.")

    return np.mean(training_accuracies), np.mean(testing_accuracies), W, b, np.mean(runtimes)
    
    
# Function to train a NN using SGD
def run_regression_sgd(num_experiments, sample_size, hidden_layers, random_nb):
    testing_accuracies = []

    for i in range(num_experiments):
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data_regression(sample_size, random_nb+i*sample_size)
        
        # Train the model using SGDRegressor
        model_sgd = SGDRegressor(max_iter=1000, tol=1e-3)
        model_sgd.fit(X_train, y_train)

        # Make predictions
        y_pred_train_sgd = model_sgd.predict(X_train)
        y_pred_test_sgd = model_sgd.predict(X_test)

        # Evaluate performance
        mse_train_sgd = mean_squared_error(y_train, y_pred_train_sgd)
        mse_test_sgd = mean_squared_error(y_test, y_pred_test_sgd)
        testing_accuracies.append(mse_test_sgd)
        print("SGD Regressor")
        print(f"Train Mean Squared Error: {mse_train_sgd:.2f}")
        print(f'Test Mean Squared Error: {mse_test_sgd:.2f}\n')

    return np.mean(testing_accuracies)

########################################################

def main():
    num_experiments = 1
    random_nb = 52 #np.random.randint(390)
    print(random_nb)
    sample_size = 9  # Number of samples to use for training
    hidden_layers = [8]  # Example hidden layers
    lambda_reg = 0.01  # Regularization parameter
    current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _ = run_regression_mip(current_date_time, num_experiments, sample_size, hidden_layers, random_nb, lambda_reg = 0.0, warm_start = False, W_init = None, b_init = None)
    _ = run_regression_sgd(num_experiments, sample_size, hidden_layers, random_nb)
    return

if __name__ == '__main__':
    main()