from experiments import run_multiple_experiments_warm_start
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import os

# Define parameters
num_experiments = 1
sample_size = 5
hidden_layers = [2]  # Hidden layers configuration
M = [17, 17 * hidden_layers[0] + 1]
margin = M[-1] * 0.1  # A reasonable margin (for SAT margin) should be a small fraction of this estimated output range
epsilon = 1.0e-1  # set the precision
lambda_reg = [0.0, 1e-3, 1e-1, 1.0]
print(lambda_reg)
dataset = 'smaller'
loss_function = 'hinge'  # Choose between 'max_correct', 'hinge', or 'sat_margin'
warm_start = False

accuracy_train_list = [[] for i in range(len(lambda_reg))]
accuracy_test_list = [[] for i in range(len(lambda_reg))]
runtime_list = [[] for i in range(len(lambda_reg))]
W_init, b_init = None, None

current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
random_nb = np.random.randint(1000)
print(random_nb)

for i, param in enumerate(lambda_reg):
    average_accuracy_train, accuracy_test, W_opt, b_opt, average_runtime = run_multiple_experiments_warm_start(
        current_date_time, num_experiments, sample_size, hidden_layers, M, margin, epsilon, loss_function, random_nb, 
        param, warm_start, W_init, b_init, dataset)
    accuracy_train_list[i].append(average_accuracy_train)
    accuracy_test_list[i].append(accuracy_test)
    runtime_list[i].append(average_runtime)
    W_init = W_opt
    b_init = b_opt


# Create figure and twin axes
fig, ax1 = plt.subplots(figsize=(10, 8))

# Plot accuracy on the left y-axis
color = 'tab:red'
ax1.set_xlabel('Lambda')
ax1.set_ylabel('Accuracy', color=color)
for i, param in enumerate(lambda_reg):
    #ax1.scatter([param] * num_experiments, accuracy_train_list[i], color=color, marker='o', label='Training Accuracy' if i == 0 else "")
    ax1.scatter([i] * num_experiments, accuracy_test_list[i], color=color, marker='x', label='Test Accuracy' if i == 0 else "")
ax1.set_xticks(range(len(lambda_reg)))  # Set x-ticks to the number of lambda values
ax1.set_xticklabels(lambda_reg)  # Label the x-ticks with the lambda values
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for computation time
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Computation Time (s)', color=color)
for i, param in enumerate(lambda_reg):
    ax2.scatter([i] * num_experiments, runtime_list[i], color=color, marker='o', label='Computation Time' if i == 0 else "")
ax2.set_xticks(range(len(lambda_reg)))  # Set x-ticks to the number of lambda values
ax2.set_xticklabels(lambda_reg)  # Label the x-ticks with the lambda values
ax2.tick_params(axis='y', labelcolor=color)

# Add title and descriptive text
plt.title('Accuracy and Computation Time for different values of Lambda')
fig.subplots_adjust(bottom=0.2)  # Increase bottom space
description_text = (
    f"Configuration :\n"
    f"Loss Function: {loss_function}\n"
    f"Neural Network Structure: { hidden_layers}\n"
    f"Regularization Parameter: {lambda_reg}\n"
    f"Big M Parameters : {M}\n"
    f"Warm Start: {warm_start}\n"
    f"Retrieve training data from: {random_nb}\n"
    f"Datasets: {dataset}"
)
plt.figtext(0.5, 0.14, description_text, ha='center', va='top', fontsize=7, wrap=True)

# Save the figure with a meaningful name
directory = f'graphs/regularisation/{loss_function}/{random_nb}/{current_date_time}'
if not os.path.exists(directory):
    os.makedirs(directory)
file_name = 'accuracy_and_time_vs_lambda_reg.png'
full_path = os.path.join(directory, file_name)
plt.savefig(full_path)

#plt.show()