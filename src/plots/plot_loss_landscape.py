import sys, os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from classification import predict_with_mip, get_W_b_opt

# Function to find non-zero weights with layer information
def find_non_zeros_non_ones(W):
    non_zero_indices = []
    for layer_idx, W_layer in enumerate(W):
        indices = np.argwhere((W_layer != 0.0) & (W_layer != 1.0) & (W_layer != -1.0))
        for idx in indices:
            non_zero_indices.append((layer_idx, *idx))
    return non_zero_indices

# Function to compute the hinge loss function value
def hinge_loss(W, b, X, y_one_hot):
    # Predict function assuming linear model
    predictions = predict_with_mip(W, b, X)

    # Hine loss computation
    hinge_loss = 0
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            y_true = 2 * y_one_hot[i,j] - 1
            hinge_loss += np.maximum(0, (1 - y_true * predictions[i][j])**2)
        
    return hinge_loss

def hinge_loss_and_L1_regularization(W, b, X, y_one_hot, lambda_reg):    
    # Hinge loss computation
    hinge_loss_value = hinge_loss(W, b, X, y_one_hot)
    
    # L1 regularization term
    reg_term = 0.0
    for layer_weight_matrix in W:
        for layer_weight in layer_weight_matrix:
            for w in layer_weight: 
                reg_term += np.abs(w)
    for layer_bias in b :
        for bias in layer_bias:
            reg_term += np.abs(bias)
    l1_regularization = lambda_reg * reg_term
    
    return hinge_loss_value + l1_regularization


########################################################


# Define parameters
sample_size = 10
dataset = 'mnist'
loss_function = 'hinge'  # Choose between 'max_correct', 'hinge', or 'sat_margin'
hidden_layers = [4]  # Hidden layers configuration
M = [785, 785 * hidden_layers[0] + 1]
margin = M[-1] * 0.1  # A reasonable margin (for SAT margin) should be a small fraction of this estimated output range
epsilon = 1.0e-1  # set the precision
lambda_reg = 0.0
warm_start = False
current_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
random_nb = np.random.randint(1000)
print(random_nb)

# Run experiments with each loss function
W_opt, b_opt, X_train_sample, y_train_sample, y_train_sample_one_hot = get_W_b_opt(
    current_date_time, sample_size, hidden_layers, M, margin, epsilon, loss_function, random_nb, 
lambda_reg, warm_start, None, None, dataset)

# Find non-zero weights across all layers
non_zero_indices_all = find_non_zeros_non_ones(W_opt)

# Select two non-zero weights randomly
if len(non_zero_indices_all) < 2:
    raise ValueError("Not enough non-zero weights to select from.")
selected_indices = np.random.choice(len(non_zero_indices_all), 2, replace=False)
weight_index1 = non_zero_indices_all[selected_indices[0]]
weight_index2 = non_zero_indices_all[selected_indices[1]]

# Define a grid around the optimal values of these weights
delta = 0.3  # range to vary the weights around the optimal value
num_points = 50  # number of points in the grid
W1_opt_val = W_opt[weight_index1[0]][weight_index1[1], weight_index1[2]]
W2_opt_val = W_opt[weight_index2[0]][weight_index2[1], weight_index2[2]]
print(f'Optimal weight picked at index({weight_index1[0]}, {weight_index1[1]}, {weight_index1[2]}) = {W1_opt_val} ')
print(f'Optimal weight picked at index({weight_index2[0]}, {weight_index2[1]}, {weight_index2[2]}) = {W2_opt_val} ')
W1_range = np.linspace(W1_opt_val - delta, W1_opt_val + delta, num_points)
W2_range = np.linspace(W2_opt_val - delta, W2_opt_val + delta, num_points)

# Initialize the loss landscape
loss_landscape = np.zeros((num_points, num_points))

# Compute the loss for each point in the grid
for i, W1 in enumerate(W1_range):
    for j, W2 in enumerate(W2_range):
        W_varied = [W.copy() for W in W_opt]
        W_varied[weight_index1[0]][weight_index1[1], weight_index1[2]] = W1
        W_varied[weight_index2[0]][weight_index2[1], weight_index2[2]] = W2
        loss_landscape[i, j] = hinge_loss_and_L1_regularization(W_varied, b_opt, X_train_sample, y_train_sample_one_hot, lambda_reg)

# Plot the 3D surface plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(W1_range, W2_range)
ax.plot_surface(X, Y, loss_landscape, cmap='viridis')
ax.set_xlabel(f'Weight {weight_index1}')
ax.set_ylabel(f'Weight {weight_index2}')
ax.set_zlabel('Loss')
ax.set_title(f'3D Loss Landscape (Hinge Loss + L1 Regularization with lambda = {lambda_reg})')

# Adding descriptive text below the plot
plt.subplots_adjust(bottom=0.2)  # Increase bottom space
description_text = (
    f"Configuration :\n"
    f"Loss Functions: {loss_function}\n"
    f"Neural Network Structure: {hidden_layers}\n"
    f"Regularization Parameter: {lambda_reg}\n"
    f"Big M Parameters : {M}\n"
    f"Warm Start: {warm_start}\n"
    f"Retrieve {sample_size} training points from: {random_nb} \n"
    f"Datasets: {dataset}"
)

# Adding text
plt.figtext(0.5, 0.14, description_text, ha='center', va='top', fontsize=7, wrap=True)

# Save the figure with a meaningful name
directory = f'graphs/loss_landscape/{random_nb}/reg{lambda_reg}/warmstart_{warm_start}/{current_date_time}'
file_name = '3D_plot.png'
full_path = os.path.join(directory, file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(full_path)

#plt.show()

# Plot the heatmap with contour lines
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, loss_landscape, levels=50, cmap='viridis')
plt.colorbar(label='Loss')
plt.contour(X, Y, loss_landscape, levels=50, colors='k', linewidths=0.5)
plt.xlabel(f'Weight {weight_index1}')
plt.ylabel(f'Weight {weight_index2}')
plt.title(f'Loss Landscape (Hinge Loss + L1 Regularization with lambda = {lambda_reg})')

# Adding descriptive text below the plot
plt.subplots_adjust(bottom=0.2)  # Increase bottom space

# Adding text
plt.figtext(0.5, 0.14, description_text, ha='center', va='top', fontsize=7, wrap=True)

# Save the figure with a meaningful name
directory = f'graphs/loss_landscape/{random_nb}/reg{lambda_reg}/warmstart_{warm_start}/{current_date_time}'
file_name = '2D_plot.png'
full_path = os.path.join(directory, file_name)
plt.savefig(full_path)

#plt.show()
