from experiments import run_multiple_experiments_warm_start
from matplotlib import pyplot as plt
from datetime import datetime

# Define parameters
num_experiments = 1
hidden_layers = [32]  # Hidden layers configuration
M = 2.6e4   # Big M constant for ReLU activation constraints (output range)
margin = 300    # A reasonable margin (for SAT margin) should be a small fraction of this estimated output range
epsilon = 1.0e-6    # set the precision
loss_function = 'hinge'  # Choose between 'max_correct', 'hinge', or 'sat_margin'

size_list = []
accuracy_train_list = []
accuracy_test_list = []

for size in range(10, 30, 10):
    sample_size = size  # number of data points
    size_list.append(sample_size)
    # Run the experiments and calculate the average accuracy
    average_accuracy_train, accuracy_test = run_multiple_experiments_warm_start(num_experiments, sample_size, hidden_layers, M, margin, epsilon, loss_function, True)
    accuracy_train_list.append(average_accuracy_train)
    accuracy_test_list.append(accuracy_test)
    print(f"Average accuracy over {num_experiments} experiments with {sample_size} training points: {average_accuracy_train}")


# Plotting the graph
plt.figure(figsize=(10, 6))
plt.scatter(size_list, accuracy_train_list, color='r', label=f'Training Accuracy') #over {num_experiments} simulations')
plt.scatter(size_list, accuracy_test_list, color='b', label='Test Accuracy')
plt.title(f'Training and test accuracy depending on the training set size using {loss_function} loss function')
plt.xlabel('Training Set Size')
plt.ylabel(f'Accuracy')

# Save the figure with a meaningful name
current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
file_name = f'training_test_accuracy_{loss_function}_loss_{current_date_time}.png'
plt.savefig(file_name)
