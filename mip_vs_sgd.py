from experiments import run_multiple_experiments_warm_start, run_experiments_with_sgd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import os

# Define parameters
num_experiments = 1
hidden_layers = [32]  # Hidden layers configuration
M = [785, 785 * hidden_layers[0] + 1]
margin = M[-1] * 0.01  # A reasonable margin (for SAT margin) should be a small fraction of this estimated output range
epsilon = 1.0e-1  # set the precision
lambda_reg = 0.0
dataset = 'mnist'
loss_function = 'sat_margin'  # Choose between 'max_correct', 'hinge', or 'sat_margin'
warm_start = False

size_list = []
accuracy_train_list = [[], []]
accuracy_test_list = [[], []]
current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
random_nb = np.random.randint(1000)
print(random_nb)


for size in range(9, 28, 3):
    sample_size = size  # number of data points
    size_list.append(sample_size)
    
    # Run experiments with each loss function
    average_accuracy_train, accuracy_test, W_opt, b_opt, average_runtime = run_multiple_experiments_warm_start(
        current_date_time, num_experiments, sample_size, hidden_layers, M, margin, epsilon, loss_function, random_nb, 
    lambda_reg, warm_start, None, None, dataset)
    accuracy_train_list[0].append(average_accuracy_train)
    accuracy_test_list[0].append(accuracy_test)
    accuracy_test_list[1].append(run_experiments_with_sgd(num_experiments, sample_size, hidden_layers, loss_function,random_nb, dataset))
        
    print(f"{dataset} - Training average accuracy over {num_experiments} experiments with {sample_size} training points : {average_accuracy_train}")


# Plotting the accuracy graph
plt.figure(figsize=(10, 8))
plt.scatter(size_list, accuracy_test_list[0], color='r', label=f'as a MIP', s=20)
plt.scatter(size_list, accuracy_test_list[1], color='b', label=f'with sgd', s=20)
plt.title('Accuracy on the testing set, training using a small portion of the data')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()


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
directory = f'graphs/mip_vs_sgd/{dataset}/{random_nb}/reg{lambda_reg}/warmstart_{warm_start}/{current_date_time}'
file_name = 'mip_vs_sgd.png'
full_path = os.path.join(directory, file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(full_path)

#plt.show()