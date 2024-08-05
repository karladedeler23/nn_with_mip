from regression import run_regression_mip, run_regression_sgd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import os

# Define parameters
num_experiments = 1
random_nb = np.random.randint(390)
print(random_nb)
hidden_layers = [8]  # Example hidden layers
lambda_reg = 0.0  # Regularization parameter

size_list = []
accuracy_train_list = [[], []]
accuracy_test_list = [[], []]
current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')



for size in range(8, 19, 2):
    sample_size = size  # number of data points
    size_list.append(sample_size)
    
    # Run experiments with each loss function
    average_accuracy_train, accuracy_test, _ , _ , _ = run_regression_mip(current_date_time, num_experiments, sample_size, hidden_layers, random_nb, lambda_reg = 0.0, warm_start = False, W_init = None, b_init = None)
    accuracy_train_list[0].append(average_accuracy_train)
    accuracy_test_list[0].append(accuracy_test)
    accuracy_test_list[1].append(run_regression_sgd(num_experiments, sample_size, hidden_layers, random_nb))
        
    print(f"TRAINING - MSE over {num_experiments} experiments with {sample_size} training points : {average_accuracy_train}")
    print(f"TESTING - MSE over {num_experiments} experiments with {sample_size} training points : {accuracy_test}")


# Plotting the accuracy graph
plt.figure(figsize=(10, 8))
plt.scatter(size_list, accuracy_test_list[0], color='r', label=f'as a MIP', s=20)
plt.scatter(size_list, accuracy_test_list[1], color='b', label=f'with SGDRegressor', s=20)
plt.title('Accuracy on the testing set, training using a small portion of the data')
plt.xlabel('Training Set Size')
plt.ylabel('Mean squared Error')
plt.legend()


# Adding descriptive text below the plot
plt.subplots_adjust(bottom=0.2)  # Increase bottom space
description_text = (
    f"Configuration :\n"
    f"Loss Functions: mse \n"
    f"Neural Network Structure: {hidden_layers}\n"
    f"Regularization Parameter: {lambda_reg}\n"
    f"Retrieve training data from: {random_nb}\n"
)
# Adding text
plt.figtext(0.5, 0.14, description_text, ha='center', va='top', fontsize=7, wrap=True)

# Save the figure with a meaningful name
directory = f'graphs/mip_vs_sgd/regression/{random_nb}/reg{lambda_reg}//{current_date_time}'
file_name = 'mip_vs_sgd.png'
full_path = os.path.join(directory, file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(full_path)

#plt.show()