from experiments import run_multiple_experiments_warm_start
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import os

# Define parameters
num_experiments = 1
datasets = ['mnist', 'smaller'] 
hidden_layers = [[32], [8]]  # Hidden layers configuration
M = [[784 + 1, 785 * hidden_layers[0][0] + 1], [16 + 1, 17 * hidden_layers[1][0] + 1]] # Big M constant for ReLU activation constraints (w and b are max = 1)
margin = [M[0][-1] * 0.01, M[1][-1] * 0.1]  # A reasonable margin (for SAT margin) should be a small fraction of this estimated output range
epsilon = 1.0e-1  # set the precision
lambda_reg = 0.0
loss_function = 'hinge'  # Choose between 'max_correct', 'hinge', or 'sat_margin'
warm_start = False

size_list = []
accuracy_train_list = [[] for _ in range(len(datasets))]
accuracy_test_list = [[] for _ in range(len(datasets))]
runtime_list = [[] for _ in range(len(datasets))]
current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
random_nb = np.random.randint(1000)
print(random_nb)

W_init = [None for _ in range(len(datasets))]
b_init = [None for _ in range(len(datasets))]

for size in range(9, 20, 3):
    sample_size = size  # number of data points
    size_list.append(sample_size)
    
    for i, dataset in enumerate(datasets):
        average_accuracy_train, accuracy_test, W_opt, b_opt, average_runtime = run_multiple_experiments_warm_start(
            current_date_time, num_experiments, sample_size, hidden_layers[i], M[i], margin[i], epsilon, loss_function, random_nb, 
            lambda_reg, warm_start, W_init[i], b_init[i], dataset)
        accuracy_train_list[i].append(average_accuracy_train)
        accuracy_test_list[i].append(accuracy_test)
        runtime_list[i].append(average_runtime)
    
        W_init[i] = W_opt
        b_init[i] = b_opt

# Plotting the accuracy graph
plt.figure(figsize=(10, 8))
colors = ['pink', 'purple']
for i in range(len(datasets)):
    plt.scatter(size_list, accuracy_train_list[i], color=colors[i], marker='o', label=f'Training Accuracy - {datasets[i]}', s=20)
    plt.scatter(size_list, accuracy_test_list[i], color=colors[i], marker='x', label=f'Test Accuracy - {datasets[i]}', s=20)
plt.title('Accuracy depending on the Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()

# Adding descriptive text below the plot
plt.subplots_adjust(bottom=0.2)  # Increase bottom space
description_text = (
    f"Configuration :\n"
    f"Loss Function: {loss_function}\n"
    f"Neural Network Structure: {hidden_layers}\n"
    f"Regularization Parameter: {lambda_reg}\n"
    f"Big M Parameters : {M}\n"
    f"Warm Start: {warm_start}\n"
    f"Retrieve training data from: {random_nb}\n"
    f"Datasets: {datasets[0]} and {datasets[1]}"
)
# Adding text
plt.figtext(0.5, 0.14, description_text, ha='center', va='top', fontsize=7, wrap=True)

# Save the figure with a meaningful name
directory = f'graphs/perf_datasets/{loss_function}/{random_nb}/reg{lambda_reg}/warmstart_{warm_start}/{current_date_time}'
file_name = 'accuracy_mnist_vs_smaller.png'
full_path = os.path.join(directory, file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(full_path)

# Plotting the computation time graph
plt.figure(figsize=(10, 8))
for i in range(len(datasets)):
    plt.plot(size_list, runtime_list[i], color=colors[i], markersize=5, label=f'Computation Time - {datasets[i]}')
plt.title('Computation Time depending on the Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Computation Time (s)')
plt.legend()

# Adding descriptive text below the plot
plt.subplots_adjust(bottom=0.2)  # Increase bottom space

# Adding text
plt.figtext(0.5, 0.14, description_text, ha='center', va='top', fontsize=7, wrap=True)

# Save the figure with a meaningful name
file_name_time = 'computation_time_mnist_vs_smaller.png'
full_path_time = os.path.join(directory, file_name_time)
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(full_path_time)

#plt.show()