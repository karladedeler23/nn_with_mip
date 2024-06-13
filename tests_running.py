from experiments import run_multiple_experiments

# Define parameters
num_experiments = 10
hidden_layers = [32]  # Hidden layers configuration
sample_size = 10  # number of data points
M = 2.6e4   # Big M constant for ReLU activation constraints (output range)
margin = 260    # A reasonable margin (for SAT margin) should be a small fraction of this estimated output range
epsilon = 1.0e-6    # set the precision
loss_function = 'hinge'  # Choose between 'max_correct', 'hinge', or 'sat_margin'

# Run the experiments and calculate the average accuracy
average_accuracy = run_multiple_experiments(num_experiments, sample_size, hidden_layers, M, margin, epsilon, loss_function)
print(f"Average Accuracy over {num_experiments} experiments: {average_accuracy}")