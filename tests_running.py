from experiments import run_multiple_experiments_warm_start
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np

# Define parameters
num_experiments = 1
hidden_layers = [4]  # Hidden layers configuration
M = [16, 16*hidden_layers[0]] # Big M constant for ReLU activation constraints (w and b are max = 1)
margin = M[-1]*0.1     # A reasonable margin (for SAT margin) should be a small fraction of this estimated output range
epsilon = 1.0e-4    # set the precision
lambda_reg = 1.0e-4
loss_function = 'hinge'  # Choose between 'max_correct', 'hinge', or 'sat_margin'
warm_start = True

size_list = []
accuracy_train_list = []
accuracy_test_list = []
random_nb = np.random.randint(1000)
print(random_nb)

W_init, b_init = None, None
for size in range(7, 58, 5):
    sample_size = size  # number of data points
    size_list.append(sample_size)
    # Run the experiments and calculate the average accuracy
    average_accuracy_train, accuracy_test, W_opt, b_opt = run_multiple_experiments_warm_start(num_experiments, sample_size, hidden_layers, M, margin, epsilon, loss_function, random_nb, lambda_reg, warm_start, W_init, b_init)    
    accuracy_train_list.append(average_accuracy_train)
    accuracy_test_list.append(accuracy_test)
    print(f"Average accuracy over {num_experiments} experiments with {sample_size} training points: {average_accuracy_train}")
    W_init = W_opt
    b_init = b_opt

# Plotting the graph
plt.figure(figsize=(10, 8))
plt.scatter(size_list, accuracy_train_list, color='r', label=f'Training Accuracy') #over {num_experiments} simulations')
plt.scatter(size_list, accuracy_test_list, color='b', label='Test Accuracy')
plt.title(f'Accuracy depending on the Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel(f'Accuracy')
plt.legend()

# Adding descriptive text below the plot
plt.subplots_adjust(bottom=0.2)  # Increase bottom space
description_text = (
    f"Configuration :\n"
    f"Loss Function: {loss_function}\n"
    f"Neural Network Structure: {[16] + hidden_layers + [10]}\n"
    f"Regularization Parameter: {lambda_reg}\n"
    f"Big M Parameters : {M}\n"
    f"Warm Start: {warm_start}\n"
    f"Retrieve training data from: {random_nb}"
)
# Adding text
plt.figtext(0.5, 0.14, description_text, ha='center', va='top', fontsize=9, wrap=True)

# Save the figure with a meaningful name
current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
file_name = f'pen_training_test_accuracy_{loss_function}_loss_{current_date_time}.png'
plt.savefig(file_name)