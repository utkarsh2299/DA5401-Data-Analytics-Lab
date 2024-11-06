import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#==================================================================================================
def load_and_preprocess_data(file_path, matrix_size=1000, range_min=0, range_max=100):
    
    df = pd.read_csv(file_path)
  
    df.columns = ['x', 'y']
    
    # Normalize the data to a 0-100 range and then scale to the matrix size
    df['x_normalized'] = ((df['x'] - range_min) / (range_max - range_min)) * (matrix_size - 1)
    df['y_normalized'] = ((df['y'] - range_min) / (range_max - range_min)) * (matrix_size - 1)
    
    # Discretize by converting the normalized values to integer indices
    df['x_discrete'] = df['x_normalized'].round().astype(int)
    df['y_discrete'] = df['y_normalized'].round().astype(int)
    # print(df['x_discrete'])
    # print(df['x'])
    return df, matrix_size

def create_boolean_matrix(df, matrix_size):
    # Create a boolean matrix initialized with False
    sparse_matrix = np.zeros((matrix_size, matrix_size), dtype=bool)
    
    # Populate the matrix with True where the data points exist
    sparse_matrix[df['x_discrete'], df['y_discrete']] = True
    
    return sparse_matrix

def rotate_matrix(matrix):
    # Rotate the matrix 90 degrees clockwise
    rotated_matrix = matrix[::-1, :]
    
    return rotated_matrix

def flip(matrix):
    # Flip the matrix vertically  and transpose
    flipped_matrix = matrix[:, ::-1].T
    
    return flipped_matrix

def plot_matrix(matrix, title):
    # Convert to integer matrix for visualization
    matrix_int = matrix.astype(int)
    
    # print(matrix_int.shape)

    plt.figure(figsize=(15, 5))
    plt.imshow(matrix_int, cmap='CMRmap', origin='lower')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(title)
    plt.show()

def plot_scatter(df, matrix_size, title):
    
    plt.figure(figsize=(10, 10))
    plt.scatter(df['x_discrete'], df['y_discrete'], color='blue', marker='o', alpha=0.6)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(title)
    plt.xlim(0, matrix_size - 1)
    plt.ylim(0, matrix_size - 1)
    plt.grid(True)
    plt.show()

def plot_scatter_from_matrix(matrix, matrix_size, title):
    # Find the indices of the True values in the matrix 
    #convert the sparse matrices into their respective X-Y coordinates.
    y_indices, x_indices = np.where(matrix)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(x_indices, y_indices, color='hotpink', marker='o', alpha=0.6)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(title)
    plt.xlim(0, matrix_size - 1)
    plt.ylim(0, matrix_size - 1)
    plt.grid(True)
    plt.show()

# ===================================================================================================
file_path = 'data.csv'
matrix_size = 1000

df, matrix_size = load_and_preprocess_data(file_path, matrix_size)

sparse_matrix = create_boolean_matrix(df, matrix_size)

rotated_matrix = rotate_matrix(sparse_matrix)

flipped_matrix = flip(sparse_matrix)

# Plots
plot_matrix(sparse_matrix.T, "Original Matrix")
plot_matrix(rotated_matrix, "Rotated Matrix (90 degrees clockwise)")
plot_matrix(flipped_matrix, "Flipped Matrix (Vertically)")

# Scatter plots
plot_scatter(df, matrix_size, "Scatter Plot of Original Data")
plot_scatter_from_matrix(rotated_matrix, matrix_size, "Scatter Plot of Rotated Matrix (90 degrees clockwise)")
plot_scatter_from_matrix(flipped_matrix, matrix_size, "Scatter Plot of Flipped Matrix (Vertically)")
