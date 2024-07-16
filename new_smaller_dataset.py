import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(n, random_nb):
    # Load the Pen-Based Recognition of Handwritten Digits dataset from UCI repository
    url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra'
    url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes'

    # Read the CSV files
    train_data = pd.read_csv(url_train, header=None)
    test_data = pd.read_csv(url_test, header=None)

    # Split into features and labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Select one data point per class for training sample
    selected_indices = []
    for i in range(n):  # Iterate through the dataset to select one data point per class
        class_indices = np.where(y_train == (i % 10))[0]
        if len(class_indices) > random_nb + i:
            index = class_indices[random_nb + i]  # Get the index of one of the occurrences of the class
        else:
            index = class_indices[0]  # In case the desired index is out of bounds, use the first index
        selected_indices.append(index)
    X_train_sample = X_train[selected_indices]
    y_train_sample = y_train[selected_indices]

    # Normalize the inputs
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train_sample = scaler.transform(X_train_sample)
    X_test = scaler.transform(X_test)

    # Convert to "one-hot" vectors using the to_categorical function
    num_classes = 10
    y_train_one_hot = to_categorical(y_train, num_classes)
    y_train_sample_one_hot = to_categorical(y_train_sample, num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes)

    return (X_train_sample, y_train_sample, y_train_sample_one_hot), (X_test, y_test, y_test_one_hot), (X_train, y_train, y_train_one_hot)

# Example usage
n = 10
random_nb = 0
(X_train_sample, y_train_sample, y_train_sample_one_hot), (X_test, y_test, y_test_one_hot), (X_train, y_train, y_train_one_hot) = load_and_preprocess_data(n, random_nb)

print(X_train_sample.shape)
print(y_train_sample_one_hot.shape)