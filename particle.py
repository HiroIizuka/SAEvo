import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict
import pickle
import os
import datetime
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import shutil
import json

'''
# Create directory for saving particles
os.makedirs('saved_particles', exist_ok=True)
# Create directory for visualizations
os.makedirs('particle_visualizations', exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
vis_dir = os.path.join('particle_visualizations', timestamp)
os.makedirs(vis_dir, exist_ok=True)

save_dir = os.path.join('saved_particles', timestamp)
os.makedirs(save_dir, exist_ok=True)

# Copy the currently running script file (particle.py) to vis_dir
current_script_path = os.path.abspath(__file__)  # Absolute path of the currently running script
script_filename = os.path.basename(current_script_path)  # Get only the filename
destination_path = os.path.join(vis_dir, script_filename)  # Destination path

try:
    shutil.copy2(current_script_path, destination_path)  # Copy file (preserving metadata)
    print(f"Script file copied to {destination_path}")
except Exception as e:
    print(f"Error occurred while copying script file: {e}")
'''

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

activation = 'tanh'

def calculate_reconstruction_error(original, reconstructed):
    # Calculate reconstruction error
    return torch.mean((original - reconstructed) ** 2).item()

def calculate_min_reconstruction_error(original, reconstructed):
    original_expanded = original.expand_as(reconstructed)
    distance = torch.mean((original_expanded - reconstructed) ** 2, dim=1)
    min_error,min_index = torch.min(distance, dim=0)
    return min_error.item()

def compute_average_reconstruction_error(x, model_forward, model_backward, sample_size=5000):
    num_particles = x.shape[0]
    # Sampling
    sample_indices = torch.randperm(num_particles)[:sample_size]
    x_sampled = x[sample_indices]
    
    # Forward transformation
    out = model_forward(x_sampled)
    out = out.view(sample_size * num_particles, 1, -1)  # Reshape
    
    sample_indices = torch.randperm(out.shape[0])[:sample_size]
    out_sampled = out[sample_indices]

    # Backward transformation
    reconstructed = model_backward(out_sampled)

    # Calculate reconstruction error
    total_error = 0.0
    for i in range(sample_size):
        total_error += calculate_min_reconstruction_error(x_sampled[i], reconstructed[i])
    
    # Average reconstruction error
    average_error = total_error / (sample_size)
    return average_error

def compute_average_reconstruction_error2(x, model_forward, model_forward2, model_backward2, model_backward, sample_size=5000):
    num_particles = x.shape[0]
    # Sampling
    sample_indices = torch.randperm(num_particles)[:sample_size]
    x_sampled = x[sample_indices]
    
    # Forward transformation
    out = model_forward(x_sampled)
    out = out.view(sample_size * num_particles, 1, -1)  # Reshape
    
    sample_indices = torch.randperm(out.shape[0])[:sample_size]
    out_sampled = out[sample_indices]

    # Forward transformation 2
    out2 = model_forward2(out_sampled)
    out2 = out2.view(sample_size * num_particles, 1, -1)  # Reshape
    
    sample_indices = torch.randperm(out2.shape[0])[:sample_size]
    out_sampled2 = out2[sample_indices]

    # Backward transformation
    reconstructed = model_backward2(out_sampled2)
    reconstructed = reconstructed.view(sample_size*num_particles, 1, -1)  # Reshape

    sample_indices = torch.randperm(reconstructed.shape[0])[:sample_size]
    reconstructed_sampled = reconstructed[sample_indices]

    # Backward transformation
    reconstructed2 = model_backward(reconstructed_sampled)

    recon = model_backward(out_sampled)

    # Calculate reconstruction error
    total_error = 0.0
    total_error2 = 0.0
    for i in range(sample_size):
        total_error += calculate_min_reconstruction_error(x_sampled[i], reconstructed2[i])
        total_error2 += calculate_min_reconstruction_error(x_sampled[i], recon[i])
    
    # Average reconstruction error
    average_error = total_error / (sample_size)
    average_error2 = total_error2 / (sample_size)

    return average_error, average_error2

def save_particles(particles_dict, iteration, save_dir, reducers=None, axis_limits=None):
    """
    Function to save particles
    
    Args:
        particles_dict: Dictionary of particles {name: tensor}
        iteration: Current iteration number
        save_dir: Save directory
        reducers: List of dimensionality reducers such as PCA
        axis_limits: List of axis ranges
    """
    # Filename for saving
    filename = os.path.join(save_dir, f'particles_iter_{iteration:04d}.pkl')
    
    # Data to save
    save_data = {}
    
    # Move particle data to CPU before saving
    for name, tensor in particles_dict.items():
        save_data[name] = tensor.cpu()
    
    # Also save dimensionality reducers and axis limits
    if reducers is not None:
        save_data['reducers'] = reducers
    
    if axis_limits is not None:
        save_data['axis_limits'] = axis_limits
    
    # Save data
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Particle state saved to file {filename}")


# Neural network model for particle data
class Conv1DNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, kernel_size=3, stride=1, padding=0, bias=False):
        super(Conv1DNet, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias

        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding, bias=bias)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
            nn.init.uniform_(self.conv.weight, 0, 1)
        elif activation == 'tanh':
            # Scaling parameter to adjust the slope of tanh
            self.slope = 2.0  # Double the slope
            self.activation = lambda x: torch.tanh(x * self.slope)
            nn.init.uniform_(self.conv.weight, -1, 1)
        elif activation == 'relu':
            self.activation = nn.ReLU()
            nn.init.uniform_(self.conv.weight, -1, 1)
        else:
            raise ValueError("Invalid activation function")

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    
    def set_weight(self, weight):
        self.conv.weight.data = weight[:,:,:self.kernel_size]
        if self.bias:
            # Bias must be 1-dimensional
            self.conv.bias.data = weight[:,0,self.kernel_size].clone()

class ConvTranspose1DNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, kernel_size=3, stride=1, padding=0, output_padding=0, bias=False):
        super(ConvTranspose1DNet, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.padding = padding
        self.output_padding = output_padding
        self.conv = nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride, padding, output_padding, bias=bias)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
            nn.init.uniform_(self.conv.weight, 0, 1)
        elif activation == 'tanh':
            self.slope = 2.0  # Double the slope
            self.activation = lambda x: torch.tanh(x * self.slope)
            nn.init.uniform_(self.conv.weight, -1, 1)
        elif activation == 'relu':
            self.activation = nn.ReLU()
            nn.init.uniform_(self.conv.weight, -1, 1)
        else:
            raise ValueError("Invalid activation function")
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    
    def set_weight(self, weight):
        self.conv.weight.data = weight[:,:,:self.kernel_size]
        if self.bias:
            # Bias must be 1-dimensional
            self.conv.bias.data = weight[:,0,self.kernel_size].clone()

def visualize_high_dim_particles_triple(x13, x7, x3, iteration, save_dir, method='pca', random_state=None):
    """
    Function to visualize high-dimensional particles (displaying 3 types of particles simultaneously)
    
    Args:
        x13: 13-dimensional particles
        x7: 7-dimensional particles
        x3: 3-dimensional particles
        iteration: Current iteration number
        save_dir: Save directory
        method: Dimensionality reduction method ('pca' or 'tsne')
        random_state: Random seed (for reproducibility)
    
    Returns:
        reducers: List of dimensionality reducers [reducer_13, reducer_7, reducer_3]
        axis_limits: Axis ranges [x13_limits, x7_limits, x3_limits]
    """
    # Move to CPU and convert to numpy array
    x13_np = x13.cpu().numpy()
    x7_np = x7.cpu().numpy()
    x3_np = x3.cpu().numpy()
    
    # Reshape
    x13_np = x13_np[:, 0, :]  # [num_particles, D1]
    x7_np = x7_np[:, 0, :]  # [num_particles, D2]
    x3_np = x3_np[:, 0, :]  # [num_particles, D3]
    
    # Sampling (to reduce processing load)
    sample_size = min(5000, x13_np.shape[0])
    sample_indices = np.random.choice(x13_np.shape[0], sample_size, replace=False)
    x13_sampled = x13_np[sample_indices, :]
    x7_sampled = x7_np[sample_indices, :]
    x3_sampled = x3_np[sample_indices, :]
    
    # Dimensionality reduction
    if method == 'pca':
        reducer_13 = PCA(n_components=3, random_state=random_state)
        reducer_7 = PCA(n_components=3, random_state=random_state)
        reducer_3 = PCA(n_components=3, random_state=random_state)
        x13_reduced = reducer_13.fit_transform(x13_sampled)
        x7_reduced = reducer_7.fit_transform(x7_sampled)
        x3_reduced = reducer_3.fit_transform(x3_sampled)
        
        # Calculate axis ranges
        x13_limits = [
            np.min(x13_reduced[:, 0]) - 0.1, np.max(x13_reduced[:, 0]) + 0.1,
            np.min(x13_reduced[:, 1]) - 0.1, np.max(x13_reduced[:, 1]) + 0.1,
            np.min(x13_reduced[:, 2]) - 0.1, np.max(x13_reduced[:, 2]) + 0.1
        ]
        x7_limits = [
            np.min(x7_reduced[:, 0]) - 0.1, np.max(x7_reduced[:, 0]) + 0.1,
            np.min(x7_reduced[:, 1]) - 0.1, np.max(x7_reduced[:, 1]) + 0.1,
            np.min(x7_reduced[:, 2]) - 0.1, np.max(x7_reduced[:, 2]) + 0.1
        ]
        x3_limits = [
            np.min(x3_reduced[:, 0]) - 0.1, np.max(x3_reduced[:, 0]) + 0.1,
            np.min(x3_reduced[:, 1]) - 0.1, np.max(x3_reduced[:, 1]) + 0.1,
            np.min(x3_reduced[:, 2]) - 0.1, np.max(x3_reduced[:, 2]) + 0.1
        ]
         # Set dimensionality reducers and axis ranges
        reducers = [reducer_13, reducer_7, reducer_3]
        axis_limits = [x13_limits, x7_limits, x3_limits]       
    elif method == 'tsne':
        # t-SNE does not maintain state, so recalculate each time
        reducer_13 = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=random_state)
        reducer_7 = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=random_state)
        reducer_3 = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=random_state)
        x13_reduced = reducer_13.fit_transform(x13_sampled)
        x7_reduced = reducer_7.fit_transform(x7_sampled)
        x3_reduced = reducer_3.fit_transform(x3_sampled)
        
        # t-SNE does not maintain state, so return None
        reducers = None
        axis_limits = None
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(24, 8))
    
    # Plot x1
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(x13_reduced[:, 0], x13_reduced[:, 1], x13_reduced[:, 2], 
               c='b', marker='o', s=10, alpha=0.6)
    
    # Plot x2
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(x7_reduced[:, 0], x7_reduced[:, 1], x7_reduced[:, 2], 
               c='r', marker='o', s=10, alpha=0.6)
    
    # Plot x3
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(x3_reduced[:, 0], x3_reduced[:, 1], x3_reduced[:, 2], 
               c='g', marker='o', s=10, alpha=0.6)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(save_dir, f'triple_{method}_particles_iter_{iteration:04d}.png'))
    plt.close(fig)
    
    return reducers, axis_limits

def initialize_particles(tensor, activation, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        if activation == 'sigmoid':
            min_val = 0
            max_val = 1
        elif activation == 'tanh' or activation == 'relu':
            min_val = -1
            max_val = 1
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    nn.init.uniform_(tensor, min_val, max_val)
    return tensor

def get_valid_indices(out):
    cond1 = ((out > -0.1) & (out < 0.1)).all(dim=2)
    invalid_mask = cond1
    valid_mask = ~invalid_mask
    return torch.where(valid_mask)[0]

def update_particles(x, out, keep_rate, device, activation):
    num_particles = x.shape[0]
    
    # Get indices of valid particles
    valid_indices = get_valid_indices(out)
    # Initialize new particle array
    new_x = torch.zeros_like(x)
    keep_count = int(num_particles * keep_rate)

    # Randomly select and keep from existing particles
    if keep_count > 0:
        indices_to_keep = torch.randperm(num_particles)[:keep_count]
        for i, idx in enumerate(indices_to_keep):
            new_x[i] = x[idx]
     
    # Replace remaining particles with new particles
    for i in range(keep_count, num_particles):
        # If valid particles exist, select from them
        if len(valid_indices) > 0:
            random_idx = torch.randint(0, len(valid_indices), (1,), device=device)
            selected_idx = valid_indices[random_idx]
            new_x[i] = out[selected_idx]
        else:
            # If no valid particles exist, generate new particles
            if activation == 'sigmoid':
                new_x[i] = torch.rand_like(x[0])
            elif activation == 'tanh' or activation == 'relu':
                new_x[i] = torch.rand_like(x[0]) * 2 - 1
    
    return new_x

def visualize_high_dim_particles_triple_fixed_axes(x13, x7, x3, iteration, save_dir, method='pca', fixed_reducers=None, fix_axes_after=None, axis_limits=None, random_state=None):
    """
    Function to visualize high-dimensional particles (displaying 3 types of particles simultaneously, with fixed axes option)
    
    Args:
        x13: 13-dimensional particles
        x7: 7-dimensional particles
        x3: 3-dimensional particles
        iteration: Current iteration number
        save_dir: Save directory
        method: Dimensionality reduction method ('pca' or 'tsne')
        fixed_reducers: List of fixed dimensionality reducers [reducer_13, reducer_7, reducer_3]
        fix_axes_after: Fix axes after this iteration number
        axis_limits: Axis ranges [x13_limits, x7_limits, x3_limits]
        random_state: Random seed (for reproducibility)
    
    Returns:
        new_reducers: Updated list of dimensionality reducers
        new_axis_limits: Updated axis ranges
    """
    # Move to CPU and convert to numpy array
    x13_np = x13.cpu().numpy()
    x7_np = x7.cpu().numpy()
    x3_np = x3.cpu().numpy()
    
    # Reshape
    x13_np = x13_np[:, 0, :]  # [num_particles, D1]
    x7_np = x7_np[:, 0, :]  # [num_particles, D2]
    x3_np = x3_np[:, 0, :]  # [num_particles, D3]
    
    # Sampling (to reduce processing load)
    sample_size = min(5000, x13_np.shape[0])
    sample_indices = np.random.choice(x13_np.shape[0], sample_size, replace=False)
    x13_sampled = x13_np[sample_indices, :]
    x7_sampled = x7_np[sample_indices, :]
    x3_sampled = x3_np[sample_indices, :]

    print( x13_sampled.shape, x7_sampled.shape, x3_sampled.shape)
    
    # Initialize or reuse dimensionality reducers
    if fixed_reducers is None or iteration <= fix_axes_after:
        # Create new dimensionality reducers
        if method == 'pca':
            reducer_13 = PCA(n_components=3, random_state=random_state)
            reducer_7 = PCA(n_components=3, random_state=random_state)
            reducer_3 = PCA(n_components=3, random_state=random_state)
            x13_reduced = reducer_13.fit_transform(x13_sampled)
            x7_reduced = reducer_7.fit_transform(x7_sampled)
            x3_reduced = reducer_3.fit_transform(x3_sampled)
            
            # Return new reducers (for future use)
            new_reducers = [reducer_13, reducer_7, reducer_3]
            
            # Calculate axis ranges (for first time or recalculation)
            x13_limits = [
                np.min(x13_reduced[:, 0]) - 0.1, np.max(x13_reduced[:, 0]) + 0.1,
                np.min(x13_reduced[:, 1]) - 0.1, np.max(x13_reduced[:, 1]) + 0.1,
                np.min(x13_reduced[:, 2]) - 0.1, np.max(x13_reduced[:, 2]) + 0.1
            ]
            x7_limits = [
                np.min(x7_reduced[:, 0]) - 0.1, np.max(x7_reduced[:, 0]) + 0.1,
                np.min(x7_reduced[:, 1]) - 0.1, np.max(x7_reduced[:, 1]) + 0.1,
                np.min(x7_reduced[:, 2]) - 0.1, np.max(x7_reduced[:, 2]) + 0.1
            ]
            x3_limits = [
                np.min(x3_reduced[:, 0]) - 0.1, np.max(x3_reduced[:, 0]) + 0.1,
                np.min(x3_reduced[:, 1]) - 0.1, np.max(x3_reduced[:, 1]) + 0.1,
                np.min(x3_reduced[:, 2]) - 0.1, np.max(x3_reduced[:, 2]) + 0.1
            ]
            new_axis_limits = [x13_limits, x7_limits, x3_limits]
        elif method == 'tsne':
            # t-SNE does not maintain state, so recalculate each time
            reducer_13 = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=random_state)
            reducer_7 = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=random_state)
            reducer_3 = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=random_state)
            x13_reduced = reducer_13.fit_transform(x13_sampled)
            x7_reduced = reducer_7.fit_transform(x7_sampled)
            x3_reduced = reducer_3.fit_transform(x3_sampled)
            
            # t-SNE does not maintain state, so return None
            new_reducers = None
            new_axis_limits = None
    else:
        # Use existing dimensionality reducers and axis ranges
        if method == 'pca':
            reducer_13, reducer_7, reducer_3 = fixed_reducers
            # Use only transform (do not fit)
            x13_reduced = reducer_13.transform(x13_sampled)
            x7_reduced = reducer_7.transform(x7_sampled)
            x3_reduced = reducer_3.transform(x3_sampled)
            new_reducers = fixed_reducers  # Return the same reducers
            new_axis_limits = axis_limits
        elif method == 'tsne':
            # t-SNE cannot maintain previous state, so recalculate each time
            # However, preprocessing with PCA to align directions is possible
            reducer_13 = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=random_state)
            reducer_7 = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=random_state)
            reducer_3 = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=random_state)
            x13_reduced = reducer_13.fit_transform(x13_sampled)
            x7_reduced = reducer_7.fit_transform(x7_sampled)
            x3_reduced = reducer_3.fit_transform(x3_sampled)
            new_reducers = None
            new_axis_limits = None
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(24, 8))
    
    # Plot x1
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(x13_reduced[:, 0], x13_reduced[:, 1], x13_reduced[:, 2], 
               c='b', marker='o', s=10, alpha=0.6)
    
    # Set axis ranges (if fixed)
    if new_axis_limits is not None and iteration > fix_axes_after:
        x13_limits = new_axis_limits[0]
        ax1.set_xlim(x13_limits[0], x13_limits[1])
        ax1.set_ylim(x13_limits[2], x13_limits[3])
        ax1.set_zlim(x13_limits[4], x13_limits[5])
    
    # Plot x2
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(x7_reduced[:, 0], x7_reduced[:, 1], x7_reduced[:, 2], 
               c='r', marker='o', s=10, alpha=0.6)
    
    if new_axis_limits is not None and iteration > fix_axes_after:
        x7_limits = new_axis_limits[1]
        ax2.set_xlim(x7_limits[0], x7_limits[1])
        ax2.set_ylim(x7_limits[2], x7_limits[3])
        ax2.set_zlim(x7_limits[4], x7_limits[5])
    
    # Plot x3
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(x3_reduced[:, 0], x3_reduced[:, 1], x3_reduced[:, 2], 
               c='g', marker='o', s=10, alpha=0.6)
    
    if new_axis_limits is not None and iteration > fix_axes_after:
        x3_limits = new_axis_limits[2]
        ax3.set_xlim(x3_limits[0], x3_limits[1])
        ax3.set_ylim(x3_limits[2], x3_limits[3])
        ax3.set_zlim(x3_limits[4], x3_limits[5])
    
    # Save
    plt.savefig(os.path.join(save_dir, f'triple_{method}_particles_iter_{iteration:04d}.png'))
    plt.close(fig)
    
    return new_reducers, new_axis_limits

def save_reconstruction_errors(errors_dict, iteration, save_dir):
    """
    Function to save reconstruction errors
    
    Args:
        errors_dict: Dictionary of reconstruction errors {name: value}
        iteration: Current iteration number
        save_dir: Save directory
    """
    # Filename for saving
    filename = os.path.join(save_dir, 'reconstruction_errors.json')
    
    # Load existing data
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {'iterations': [], 'errors': {}}
    
    # Update data
    data['iterations'].append(iteration)
    for name, value in errors_dict.items():
        if name not in data['errors']:
            data['errors'][name] = []
        data['errors'][name].append(value)
    
    # Save data
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    # Create graph
    plt.figure(figsize=(10, 6))
    for name, values in data['errors'].items():
        plt.plot(data['iterations'], values, label=name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Errors over Iterations')
    plt.legend()
    plt.grid(True)
    
    # Save graph
    plt.savefig(os.path.join(save_dir, 'reconstruction_errors.png'))
    plt.close()

# Main function
def main(seed=42):
    """
    Main function
    
    Args:
        seed: Random seed value (default: 42)
    """
    # Fix seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Random seed set to: {seed}")

    # Create directory for saving particles
    os.makedirs('saved_particles', exist_ok=True)
    # Create directory for visualizations
    os.makedirs('particle_visualizations', exist_ok=True)
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = 'alife_journal'
    vis_dir = os.path.join('particle_visualizations', timestamp)
    os.makedirs(vis_dir, exist_ok=True)

    #save_dir = os.path.join('saved_particles', timestamp)
    save_dir = os.path.join('saved_particles', 'alife_journal')
    os.makedirs(save_dir, exist_ok=True)

    # Save command line arguments to a text file
    command_line_file = os.path.join(save_dir, 'command_line.txt')
    with open(command_line_file, 'w') as f:
        f.write(' '.join(sys.argv) + '\n')
    print(f"Command line saved to {command_line_file}")

    # Copy the currently running script file (particle.py) to vis_dir
    current_script_path = os.path.abspath(__file__)  # Absolute path of the currently running script
    script_filename = os.path.basename(current_script_path)  # Get only the filename
    destination_path = os.path.join(vis_dir, script_filename)  # Destination path

    try:
        shutil.copy2(current_script_path, destination_path)  # Copy file (preserving metadata)
        print(f"Script file copied to {destination_path}")
    except Exception as e:
        print(f"Error occurred while copying script file: {e}")



    x13_length = 13
    x7_length = 7
    x3_length = 3
    x1_length = 1

    num_particles = 5000
    max_iter =1000
    noise_strength = 0.1
    keep_rate = 0.1
    # Visualization frequency
    vis_interval = 1
    save_interval = 1
    recon_interval = 5

    react_13_1 = False
    react_1_13 = False

    react_7_1 = False
    react_1_7 = False

    react_13_7 = True
    react_7_13 = True

    react_7_3 = True
    react_3_7 = True

    using_particle_13 = False
    using_particle_7 = False
    using_particle_3 = False
    using_particle_1 = False
    
    if react_13_1 or react_1_13:
        using_particle_13 = True
        using_particle_1 = True

    if react_7_1 or react_1_7:
        using_particle_7 = True
        using_particle_1 = True

    if react_13_7 or react_7_13:
        using_particle_13 = True
        using_particle_7 = True

    if react_7_3 or react_3_7:
        using_particle_7 = True
        using_particle_3 = True

    if using_particle_3:
        x3 = torch.randn(num_particles, 1, x3_length).to(device)
        initialize_particles(x3, activation)
    if using_particle_7:
        x7 = torch.randn(num_particles, 1, x7_length).to(device)
        initialize_particles(x7, activation)
    if using_particle_13:
        x13 = torch.randn(num_particles, 1, x13_length).to(device)
        initialize_particles(x13, activation)
    if using_particle_1:
        x1 = torch.randn(num_particles, 1, x1_length).to(device)
        initialize_particles(x1, activation)
    
    if react_13_1:
        model13 = Conv1DNet(output_channels=num_particles, kernel_size=x13_length, padding=0 ).to(device)
        model13.set_weight(x13)
    
    if react_1_13:
        model13back = ConvTranspose1DNet(output_channels=num_particles, kernel_size=x13_length, padding=0).to(device)
        model13back.set_weight(x13.permute(1, 0, 2))

    if react_7_1:
        model7 = Conv1DNet(output_channels=num_particles, kernel_size=x7_length, padding=0 ).to(device)
        model7.set_weight(x7)

    if react_1_7:
        model7back = ConvTranspose1DNet(output_channels=num_particles, kernel_size=x7_length, padding=0).to(device)
        model7back.set_weight(x7.permute(1, 0, 2))

    if react_13_7:
        model13_7 = Conv1DNet(output_channels=num_particles, kernel_size=x13_length, padding=3 ).to(device)
        model13_7.set_weight(x13)

    if react_7_13:
        model7_13 = ConvTranspose1DNet(output_channels=num_particles, kernel_size=x7_length, padding=0).to(device)
        model7_13.set_weight(x7.permute(1, 0, 2))

    if react_7_3 :
        model7_3 = Conv1DNet(output_channels=num_particles, kernel_size=x7_length, padding=1).to(device)
        model7_3.set_weight(x7)

    if react_3_7 :
        model3_7 = ConvTranspose1DNet(output_channels=num_particles, kernel_size=x3_length, padding=0,stride=2).to(device)
        model3_7.set_weight(x3.permute(1, 0, 2))

    # Visualize initial state
    fixed_reducers, axis_limits = visualize_high_dim_particles_triple(x13, x7, x3, 0, vis_dir, method='pca', random_state=seed)

    # Visualization settings
    fix_axes_after = 1000  # Fix axes after this iteration number
    
    # Main loop
    for step in range(max_iter):
        if step % 10 == 0:
            print(f"Iteration: {step}")
            # Check memory usage (when using CUDA)
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        with torch.no_grad():  # Disable gradient computation to reduce memory usage
            temp_out_1 = []
            temp_out_3 = []
            temp_out_7 = []
            temp_out_13 = []

            if react_13_1:
                out13_1 = model13(x13)
                out13_1 = out13_1.view(num_particles*num_particles, 1, x1_length)
                temp_out_1.append(out13_1)

            if react_7_1:
                out7_1 = model7(x7)
                out7_1 = out7_1.view(num_particles*num_particles, 1, x1_length)
                temp_out_1.append(out7_1)
            
            if react_13_7:
                out13_7 = model13_7(x13)
                out13_7 = out13_7.view(num_particles*num_particles, 1, x7_length)
                temp_out_7.append(out13_7)
            
            if react_7_13:
                out7_13 = model7_13(x7)
                out7_13 = out7_13.view(num_particles*num_particles, 1, x13_length)
                temp_out_13.append(out7_13)

            if react_1_13:
                out1_13 = model13back(x1)
                out1_13 = out1_13.view(num_particles*num_particles, 1, x13_length)
                temp_out_13.append(out1_13)

            if react_1_7:
                out1_7 = model7back(x1)
                out1_7 = out1_7.view(num_particles*num_particles, 1, x7_length)
                temp_out_7.append(out1_7)

            if react_7_3:
                out7_3 = model7_3(x7)
                out7_3 = out7_3.view(num_particles*num_particles, 1, x3_length)
                temp_out_3.append(out7_3)

            if react_3_7:
                out3_7 = model3_7(x3)
                out3_7 = out3_7.view(num_particles*num_particles, 1, x7_length)
                temp_out_7.append(out3_7)

            if using_particle_1:
                out1 = torch.cat(temp_out_1, dim=0)
                new_x1 = update_particles(x1, out1, keep_rate, device, activation)
                x1 = new_x1 + torch.randn_like(x1) * noise_strength

            if using_particle_7:
                out7 = torch.cat(temp_out_7, dim=0)
                new_x7 = update_particles(x7, out7, keep_rate, device, activation)
                x7 = new_x7 + torch.randn_like(x7) * noise_strength

            if using_particle_3:
                out3 = torch.cat(temp_out_3, dim=0)
                new_x3 = update_particles(x3, out3, keep_rate, device, activation)
                x3 = new_x3 + torch.randn_like(x3) * noise_strength

            if using_particle_13:
                out13 = torch.cat(temp_out_13, dim=0)
                new_x13 = update_particles(x13, out13, keep_rate, device, activation)
                x13 = new_x13 + torch.randn_like(x13) * noise_strength

            if step % save_interval == 0:
                particles_dict = {}
                if using_particle_1:
                    particles_dict['x1'] = new_x1
                if using_particle_3:
                    particles_dict['x3'] = new_x3
                if using_particle_7:
                    particles_dict['x7'] = new_x7
                if using_particle_13:
                    particles_dict['x13'] = new_x13
                # Also save PCA information
                save_particles(particles_dict, step, save_dir, reducers=fixed_reducers, axis_limits=axis_limits)

            if step % recon_interval == 0:
            # Calculate reconstruction error
                error_13_and_7 = compute_average_reconstruction_error(x13, model13_7, model7_13)
                error_7_and_3 = compute_average_reconstruction_error(x7, model7_3, model3_7)
                error_13_and_3, error_13_and_7_2 = compute_average_reconstruction_error2(x13, model13_7, model7_3, model3_7, model7_13)
            
                # Save reconstruction error
                errors_dict = {
                    '13->7': error_13_and_7,
                    '7->3': error_7_and_3,
                    '13->7->3->7->13': error_13_and_3
                }
                save_reconstruction_errors(errors_dict, step, vis_dir)

                print(f"error_7_and_3: {error_7_and_3}")
                print(f"error_13_and_3: {error_13_and_3}")
                print(f"error_13_and_7: {error_13_and_7}")
                print(f"error_13_and_7_2: {error_13_and_7_2}")
            if react_13_1:
                model13.set_weight(x13)
            if react_1_13:
                model13back.set_weight(x13.permute(1, 0, 2))

            if react_7_1:
                model7.set_weight(x7)
            if react_1_7:
                model7back.set_weight(x7.permute(1, 0, 2))

            if react_13_7:
                model13_7.set_weight(x13)
            if react_7_13:
                model7_13.set_weight(x7.permute(1, 0, 2))

            if react_7_3:
                model7_3.set_weight(x7)
            if react_3_7:
                model3_7.set_weight(x3.permute(1, 0, 2))

            # Visualize periodically
            if step % vis_interval == 0:
                # Visualization with fixed axes
                fixed_reducers, axis_limits = visualize_high_dim_particles_triple_fixed_axes(
                    new_x13, new_x7, new_x3, step+1, vis_dir, 
                    method='pca', 
                    fixed_reducers=fixed_reducers,
                    fix_axes_after=fix_axes_after,
                    axis_limits=axis_limits,
                    random_state=seed
                )

        # Clear cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Visualize final state
    fixed_reducers, axis_limits = visualize_high_dim_particles_triple_fixed_axes(
        new_x13, new_x7, new_x3, max_iter, vis_dir, 
        method='pca', 
        fixed_reducers=fixed_reducers,
        fix_axes_after=fix_axes_after,
        axis_limits=axis_limits,
        random_state=seed
    )

    print(f"Visualization images saved to {vis_dir}")

    # Display animation creation method in console
    print("\nAnimation creation method:")
    print("You can create an animation with the following command (FFmpeg required):")
    print(f"ffmpeg -framerate 10 -pattern_type glob -i '{vis_dir}/particles_iter_*.png' -c:v libx264 -pix_fmt yuv420p {vis_dir}/particle_animation.mp4")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Particle simulation with fixed seed for reproducibility')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    main(seed=args.seed)
