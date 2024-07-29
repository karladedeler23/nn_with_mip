from experiments import run_multiple_experiments_warm_start
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import os

# Define parameters
num_experiments = 1
hidden_layers = [16]  # Hidden layers configuration
M0 = [784, 784 * hidden_layers[0]]
M1 = [16, 16 * hidden_layers[0]]  # Big M constant for ReLU activation constraints (w and b are max = 1)
M = [M0, M1]
margin0 = M[0][-1] * 0.1  # A reasonable margin (for SAT margin) should be a small fraction of this estimated output range
margin1 = M[1][-1] * 0.1
margin = [margin0, margin1]
epsilon = 1.0e-4  # set the precision
lambda_reg = 0.0
loss_function = 'hinge'  # Choose between 'max_correct', 'hinge', or 'sat_margin'
warm_start = False

size_list = []
accuracy_train_list_1 = []
accuracy_test_list_1 = []
runtime_list_1 = []
accuracy_train_list_2 = []
accuracy_test_list_2 = []
runtime_list_2 = []
current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
random_nb = np.random.randint(1000)
print(random_nb)

W_init_1, b_init_1 = None, None
W_init_2, b_init_2 = None, None

datasets = ['mnist', 'smaller']  # Placeholder names for the two datasets

for size in range(10, 51, 10):
    sample_size = size  # number of data points
    size_list.append(sample_size)
    
    # Run experiments for the first dataset
    average_accuracy_train_1, accuracy_test_1, W_opt_1, b_opt_1, average_runtime_1 = run_multiple_experiments_warm_start(
        current_date_time, num_experiments, sample_size, hidden_layers, M[0], margin[0], epsilon, loss_function, random_nb, 
        lambda_reg, warm_start, W_init_1, b_init_1, datasets[0])
    accuracy_train_list_1.append(average_accuracy_train_1)
    accuracy_test_list_1.append(accuracy_test_1)
    runtime_list_1.append(average_runtime_1)
    
    # Run experiments for the second dataset
    average_accuracy_train_2, accuracy_test_2, W_opt_2, b_opt_2, average_runtime_2 = run_multiple_experiments_warm_start(
        current_date_time, num_experiments, sample_size, hidden_layers, M[1], margin[1], epsilon, loss_function, random_nb, 
        lambda_reg, warm_start, W_init_2, b_init_2, datasets[1])
    accuracy_train_list_2.append(average_accuracy_train_2)
    accuracy_test_list_2.append(accuracy_test_2)
    runtime_list_2.append(average_runtime_2)
    
    print(f"{datasets[0]} - Average accuracy over {num_experiments} experiments with {sample_size} training points: {average_accuracy_train_1}")
    print(f"{datasets[1]} - Average accuracy over {num_experiments} experiments with {sample_size} training points: {average_accuracy_train_2}")
    
    W_init_1 = W_opt_1
    b_init_1 = b_opt_1
    W_init_2 = W_opt_2
    b_init_2 = b_opt_2

# Plotting the accuracy graph
plt.figure(figsize=(10, 8))
plt.scatter(size_list, accuracy_train_list_1, color='b', marker='o', label=f'Training Accuracy - {datasets[0]}', s=20)
plt.scatter(size_list, accuracy_test_list_1, color='b', marker='x', label=f'Test Accuracy - {datasets[0]}', s=20)
plt.scatter(size_list, accuracy_train_list_2, color='orange', marker='o', label=f'Training Accuracy - {datasets[1]}', s=20)
plt.scatter(size_list, accuracy_test_list_2, color='orange', marker='x', label=f'Test Accuracy - {datasets[1]}', s=20)
plt.title('Accuracy depending on the Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()

# Adding descriptive text below the plot
plt.subplots_adjust(bottom=0.2)  # Increase bottom space
description_text = (
    f"Configuration :\n"
    f"Loss Function: {loss_function}\n"
    f"Neural Network Structure: { hidden_layers}\n"
    f"Regularization Parameter: {lambda_reg}\n"
    f"Big M Parameters : {M}\n"
    f"Warm Start: {warm_start}\n"
    f"Retrieve training data from: {random_nb}\n"
    f"Datasets: {datasets[0]} and {datasets[1]}"
)
# Adding text
plt.figtext(0.5, 0.14, description_text, ha='center', va='top', fontsize=7, wrap=True)

# Save the figure with a meaningful name
directory = f'graphs/pen/{loss_function}/{random_nb}/reg{lambda_reg}/warmstart_{warm_start}/{current_date_time}'
file_name = 'accuracy.png'
full_path = os.path.join(directory, file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(full_path)

# Plotting the computation time graph
plt.figure(figsize=(10, 8))
plt.plot(size_list, runtime_list_1, color='b', markersize=5, label=f'Computation Time - {datasets[0]}')
plt.plot(size_list, runtime_list_2, color='orange', markersize=5, label=f'Computation Time - {datasets[1]}')
plt.title('Computation Time depending on the Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Computation Time (s)')
plt.legend()

# Adding descriptive text below the plot
plt.subplots_adjust(bottom=0.2)  # Increase bottom space

# Adding text
plt.figtext(0.5, 0.14, description_text, ha='center', va='top', fontsize=7, wrap=True)

# Save the figure with a meaningful name
file_name_time = 'computation_time.png'
full_path_time = os.path.join(directory, file_name_time)
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(full_path_time)

#plt.show()