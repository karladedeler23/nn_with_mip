from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from classification import run_multiple_experiments_warm_start


# Define parameters
num_experiments = 1
hidden_layers = [8]  # Hidden layers configuration
M = [17, 17 * hidden_layers[0] + 1]
margin = M[-1] * 0.01  # A reasonable margin (for SAT margin) should be a small fraction of this estimated output range
epsilon = 1.0e-1  # set the precision
lambda_reg = 0.0
dataset = 'smaller'
loss_function = 'hinge'  # Choose between 'max_correct', 'hinge', or 'sat_margin'
warm_start = [False, True]

size_list = []
accuracy_train_list = [[] for i in range(len(warm_start))]
accuracy_test_list = [[] for i in range(len(warm_start))]
runtime_list = [[] for i in range(len(warm_start))]
current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
random_nb = np.random.randint(1000)
print(random_nb)

W_init = [None for i in range(len(warm_start))]
b_init = [None for i in range(len(warm_start))]


for size in range(9, 10, 3):
    sample_size = size  # number of data points
    size_list.append(sample_size)
    
    # Run experiments with each loss function
    for i, warm_up in enumerate(warm_start):
        average_accuracy_train, accuracy_test, W_opt, b_opt, average_runtime = run_multiple_experiments_warm_start(
            current_date_time, num_experiments, sample_size, hidden_layers, M, margin, epsilon, loss_function, random_nb, 
        lambda_reg, warm_up, W_init[i], b_init[i], dataset)
        accuracy_train_list[i].append(average_accuracy_train)
        accuracy_test_list[i].append(accuracy_test)
        runtime_list[i].append(average_runtime)
        
        print(f"{dataset} - Training average accuracy over {num_experiments} experiments with {sample_size} training points using warm_start = {warm_up}: {average_accuracy_train}")
        print(f"{dataset} - Testing average accuracy over {num_experiments} experiments with {sample_size} training points using warm_start = {warm_up}: {accuracy_test}")

        W_init[i] = W_opt
        b_init[i] = b_opt

# Plotting the accuracy graph
plt.figure(figsize=(10, 8))
plt.plot(size_list, np.subtract(runtime_list[1],runtime_list[0]), color='b', markersize=5, label = 'with warm start - without')
plt.title('Difference of computation time whether using the warm start technique or not ')
plt.xlabel('Training Set Size')
plt.ylabel('Computation Time (s)')
plt.legend()
'''
# Calculate the percentage difference
percentage_difference = np.divide(np.subtract(runtime_list[0], runtime_list[1]), runtime_list[0]) * 100

# Plotting the percentage difference graph
plt.figure(figsize=(10, 8))
plt.plot(size_list, percentage_difference, color='b', marker='o', markersize=5)
plt.title('Percentage Difference in Computation Time with Warm Start Technique')
plt.xlabel('Training Set Size')
plt.ylabel('Percentage Difference (%)')
plt.legend()
'''
# Adding descriptive text below the plot
plt.subplots_adjust(bottom=0.2)  # Increase bottom space
description_text = (
    f"Configuration :\n"
    f"Loss Functions: {loss_function}\n"
    f"Neural Network Structure: { hidden_layers}\n"
    f"Regularization Parameter: {lambda_reg}\n"
    f"Big M Parameters : {M}\n"
    f"Warm Start: {warm_start}\n"
    f"Retrieve training data from: {random_nb}\n"
    f"Datasets: {dataset}"
)
# Adding text
plt.figtext(0.5, 0.14, description_text, ha='center', va='top', fontsize=7, wrap=True)

# Save the figure with a meaningful name
directory = f'graphs/warm_start/{dataset}/{random_nb}/reg{lambda_reg}/warmstart_{warm_start}/{current_date_time}'
file_name = 'effect_warm_start.png'
full_path = os.path.join(directory, file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(full_path)

# Plotting the accuracy graph
plt.figure(figsize=(10, 8))
colors = ['orange', 'red']
for i in range(len(warm_start)):
    plt.scatter(size_list, accuracy_test_list[i], color=colors[i], marker='x', label=f'Using Warm Start: {warm_start[i]}', s=20)
plt.title('Accuracy depending on the Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()

plt.subplots_adjust(bottom=0.2)  # Increase bottom space
plt.figtext(0.5, 0.14, description_text, ha='center', va='top', fontsize=7, wrap=True)


# Save the figure with a meaningful name
directory = f'graphs/warm_start/{dataset}/{random_nb}/reg{lambda_reg}/warmstart_{warm_start}/{current_date_time}'
file_name = 'accuracy_comp.png'
full_path = os.path.join(directory, file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(full_path)