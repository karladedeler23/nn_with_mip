from experiments import *
from isolating_behaviors_scripts.write_to import write_variables_to_file
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
from tensorflow.keras.regularizers import l2
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    M = [10*input_dim+1]
    for i in range(len(hidden_layers)):
        M.append(M[i]*hidden_layers[i]*10+1)
    return M

########################################################

### TRAINING NN WITH MIP

# Function to train Gurobi model for regression
def train_gurobi_model_regression(X_train_sample, y_train_sample, input_dim, hidden_layers, output_dim, lambda_reg, bound):
    n = X_train_sample.shape[0]
    M = compute_big_M(input_dim, hidden_layers)
    # print(M)
    model = initialize_model()

    weights, biases, hidden_vars, relu_activation, binary_vars, y_pred = create_variables(model, input_dim, hidden_layers, output_dim, n)
    modify_bounds_to_infinity(weights, biases, bound)

    if hidden_layers != []:
        add_hidden_layer_constraints(model, X_train_sample, weights, biases, hidden_vars, relu_activation, binary_vars, input_dim, hidden_layers, M, n)
        add_output_layer_constraints(model, relu_activation, weights, biases, hidden_vars, y_pred, binary_vars, output_dim, hidden_layers, M, n)

    else:
        add_output_layer_constraints(model, relu_activation, weights, biases, X_train_sample, y_pred, binary_vars, output_dim, hidden_layers, M, n)
    set_loss_function_regression(model, weights, biases, y_pred, y_train_sample, n, input_dim, hidden_layers, output_dim, lambda_reg)

    if optimize_model(model):
        W, b = extract_weights_biases(model, weights, biases)
        write_variables_to_file(model, weights, biases, hidden_vars, binary_vars[-1], relu_activation, y_pred, hidden_layers, output_dim, n, 'variables_values.txt')
        return model.Runtime, W, b
    else:
        return None, None, None

def modify_bounds_to_infinity(weights, biases, bound):
    # Modify the bounds of the weights and biases
    for W in weights:
        for i in W.keys():
            W[i].setAttr(GRB.Attr.LB, -bound)
            W[i].setAttr(GRB.Attr.UB, bound)

    for b in biases:
        for i in b.keys():
            b[i].setAttr(GRB.Attr.LB, -bound)
            b[i].setAttr(GRB.Attr.UB, bound)


'''
# Define variables to mimic the NN
def create_variables_regression(model, input_dim, hidden_layers, output_dim, n):
    structure = hidden_layers + [output_dim]
    weights, biases, hidden_vars = [], [], []

    previous_layer_size = input_dim
    for i in range(len(structure)):
        layer_size = structure[i]
        W = model.addVars(layer_size, previous_layer_size, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"W{i+1}")
        b = model.addVars(layer_size, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"b{i+1}")
        weights.append(W)
        biases.append(b)
        previous_layer_size = layer_size

    # Define variables for the hidden outputs
    for i, layer_size in enumerate(hidden_layers):
        z_hidden = model.addVars(n, layer_size, vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY, name=f"z{i+1}")
        hidden_vars.append(z_hidden)

    # Define the output layer variables for the final activation function (here ReLU)
    y_pred = model.addVars(n, output_dim, vtype=GRB.CONTINUOUS, lb= -GRB.INFINITY, name=f"y_pred")

    return weights, biases, hidden_vars, y_pred

# Constraints for the  hidden layers
def add_layer_constraints_regression(model, X_train_sample, weights, biases, hidden_vars, input_dim, hidden_layers, n):   
    if hidden_vars != []:
        # Constraints for the first hidden layer
        for i in range(n):
            for j in range(hidden_layers[0]):
                model.addConstr(hidden_vars[0][i, j] == gp.quicksum(X_train_sample[i, k] * weights[0][j, k] for k in range(input_dim)) + biases[0][j])

        # Constraints for subsequent hidden layers
        for l in range(1, len(hidden_layers)):
            for i in range(n):
                for j in range(hidden_layers[l]):
                    model.addConstr(hidden_vars[l][i, j] == gp.quicksum(relu_activation[l-1][i, k] * weights[l][j, k] for k in range(hidden_layers[l-1])) + biases[l][j])

# Function to add constraints for the output layer for regression
def add_output_layer_constraints_regression(model, weights, biases, hidden_vars, y_pred, output_dim, hidden_layers, n):
    # Adding constraints for the output layer (regression)
    for i in range(n):
        length = None
        if hidden_layers == []:
            length = len(hidden_vars[i])
        else : 
            length = hidden_layers[-1]
        for j in range(output_dim):
            model.addConstr(
                y_pred[i, j] == gp.quicksum(weights[-1][j, k] * hidden_vars[i, k] for k in range(length)) + biases[-1][j],
                name=f"output_layer_{i}_{j}"
            )
'''

# Function to set loss function for regression
def set_loss_function_regression(model, weights, biases, y_pred, y_train_sample, n, input_dim, hidden_layers, output_dim, lambda_reg):
    # Adding loss function for regression (MSE)
    loss_expr = gp.LinExpr()
    loss_expr += 1
    for i in range(n):
        loss_expr += 1/n * (y_pred[i, 0] - y_train_sample[i]) * (y_pred[i, 0] - y_train_sample[i])
    
    # Adding L2 regularization
    if lambda_reg != 0.0:
        print("REGULARISATION L2")

        # Add the absolute values of the weights to the regularization term
        for i, weight_matrix in enumerate(weights):
            for (j, k) in weight_matrix.keys():
                loss_expr += lambda_reg * (weight_matrix[j, k] ** 2)
        
        # Add the absolute values of the biases to the regularization term
        for i, bias_matrix in enumerate(biases):
            for j in bias_matrix.keys():
                loss_expr += lambda_reg * (bias_matrix[j] ** 2)
    
    # Objective function
    model.setObjective(loss_expr, GRB.MINIMIZE)

########################################################

# Function to compute the accuracy (Mean Squared Error)
def compute_accuracy(X, y, weights, biases):
    # Predict on the test set
    y_pred = [item[0] for item in predict_with_mip(weights, biases, X, y)]
    print(y)
    # Compute Mean Squared Error
    mse = np.mean((y_pred - y)**2)
    # print(f'{y_pred} instead of {y}')
    return mse

########################################################

### MAIN FUNCTION TO RUN THE TRAINING

def run_regression_mip(current_date_time, num_experiments, sample_size, hidden_layers, random_nb, bound, lambda_reg = 0.0, warm_start = False, W_init = None, b_init = None):
    training_accuracies, testing_accuracies = [], []
    runtimes = []
    predictions = []
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

        # Train Gurobi model and get optimal weights and biases
        if warm_start and W_init is not None and b_init is not None : 
            print('warm start')
            # Using previous weights (when we are actually training with more points)
            # runtime, W, b = train_gurobi_model_regression_warm_start(X_train, y_train, input_dim, hidden_layers, 1, M, lambda_reg, W_init, b_init)
        else :
            print('no warm start')
            runtime, W, b = train_gurobi_model_regression(X_train, y_train, input_dim, hidden_layers, 1,lambda_reg, bound)
        if W is not None and b is not None:
            runtimes.append(runtime)
            print(f"Training completed in {runtime:.2f} seconds. \n")
            # Compute and print the test accuracy
            predictions.append(predict_with_mip(W, b, X_train))
            train_mse = compute_accuracy(X_train, y_train, W, b)
            training_accuracies.append(train_mse)
            test_mse = compute_accuracy(X_test, y_test, W, b)
            testing_accuracies.append(test_mse)
            
            print("MIP")
            print(f"Train Mean Squared Error: {train_mse:.2f}")
            print(f"Test Mean Squared Error: {test_mse:.2f} \n")
            
        else:
            print("Training failed.")

    return np.mean(training_accuracies), np.mean(testing_accuracies), W, b, np.mean(runtimes), predictions
    
    
# Function to train a NN using SGD
def run_regression_sgd(num_experiments, sample_size, hidden_layers, random_nb, lambda_reg, bound):
    testing_accuracies = []
    predictions = []

    for i in range(num_experiments):
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data_regression(sample_size, random_nb + i * sample_size)
        weight_constraint = ClipConstraint(min_value=-bound, max_value=bound)
        bias_constraint = ClipConstraint(min_value=-bound, max_value=bound)

        # Build the neural network model
        model_nn = Sequential()

        # Input layer
        model_nn.add(Input(shape=(X_train.shape[1],)))
        model_nn.add(Dense(hidden_layers[0], activation='relu', kernel_regularizer=l2(lambda_reg), kernel_constraint=weight_constraint, 
            bias_constraint=bias_constraint))

        # Hidden layers
        for units in hidden_layers[1:]:
            model_nn.add(Dense(units, activation='relu', kernel_regularizer=l2(lambda_reg), kernel_constraint=weight_constraint, 
            bias_constraint=bias_constraint))
        
        # Output layer
        model_nn.add(Dense(1, activation='relu', kernel_regularizer=l2(lambda_reg)))

        # Compile the model with SGD optimizer
        sgd_optimizer = SGD(learning_rate=0.01)
        model_nn.compile(optimizer=sgd_optimizer, loss='mean_squared_error')

        # Train the model
        model_nn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Make predictions
        y_pred_train_nn = model_nn.predict(X_train).flatten()
        predictions.append(y_pred_train_nn)
        y_pred_test_nn = model_nn.predict(X_test).flatten()

        # Evaluate performance
        mse_train_nn = mean_squared_error(y_train, y_pred_train_nn)
        mse_test_nn = mean_squared_error(y_test, y_pred_test_nn)
        testing_accuracies.append(mse_test_nn)
        
        print("Neural Network with SGD")
        print(f"Train Mean Squared Error: {mse_train_nn:.2f}")
        print(f"Test Mean Squared Error: {mse_test_nn:.2f}\n")
        
        

    return predictions, np.mean(testing_accuracies)

########################################################

def main():
    num_experiments = 1
    random_nb = np.random.randint(390)
    print(random_nb)
    sample_size = 10  # Number of samples to use for training
    hidden_layers = [4]  # Example hidden layers
    bound = 5
    lambda_reg = 0.0  # Regularization parameter
    current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _, _, _, _, _, y_pred_mip = run_regression_mip(current_date_time, num_experiments, sample_size, hidden_layers, random_nb, bound, lambda_reg, warm_start = False, W_init = None, b_init = None)
    y_pred_sgd, _ = run_regression_sgd(num_experiments, sample_size, hidden_layers, random_nb, lambda_reg, bound)


    # Assuming y_test is the true values, and y_pred_sgd and y_pred_mip are the predictions
    _, _, y_test, _ = load_and_preprocess_data_regression(sample_size, random_nb)

    # Convert lists to NumPy arrays
    y_test = np.squeeze(y_test)  # Remove single-dimensional entries
    y_pred_sgd = np.squeeze(y_pred_sgd)
    y_pred_mip = np.squeeze(y_pred_mip)

    mse_sgd = mean_squared_error(y_test, y_pred_sgd)
    mse_mip = mean_squared_error(y_test, y_pred_mip)
    mse_difference = mse_sgd - mse_mip

    print(f"MSE SGD: {mse_sgd:.2f}")
    print(f"MSE MIP: {mse_mip:.2f}")
    print(f"MSE Difference: {mse_difference:.2f}")

    # Define margin
    margin = 1
    # Calculate min and max values
    min_value = min(min(y_pred_sgd), min(y_pred_mip), min(y_test))
    max_value = max(max(y_pred_sgd), max(y_pred_mip), max(y_test))
    # Set figure size to be square
    plt.figure(figsize=(8, 8))
    # Plot predictions
    plt.plot(y_test, y_test, color='green', linestyle='--')
    plt.scatter(y_test, y_pred_sgd, color='blue', label='SGD Predictions', alpha=0.5)
    plt.scatter(y_test, y_pred_mip, color='red', label='MIP Predictions', alpha=0.5)
    # Set axis labels and title
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Comparison of Model Predictions')
    # Set equal scaling
    plt.gca().set_aspect('equal', adjustable='box')
    # Set axis limits with added margin
    plt.xlim([min(y_test) - margin, max(y_test) + margin])
    plt.ylim([min_value - margin, max_value + margin])
    plt.legend()
    plt.show()

    # Residual Analysis
    residual_sgd = y_test - y_pred_sgd
    residual_mip = y_test - y_pred_mip

    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, residual_sgd, color='blue', label='SGD Residuals')
    plt.scatter(y_test, residual_mip, color='red', label='MIP Residuals')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Analysis')
    plt.legend()
    plt.show()


    return

if __name__ == '__main__':
    main()