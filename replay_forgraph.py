import os
import sys
import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob
import re

def extract_iteration_from_filename(filename):
    """Function to extract iteration number from filename"""
    match = re.search(r'iter_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def visualize_particles_triple_fixed_axes(x13, x7, x3, iteration, save_dir, fixed_reducers=None, axis_limits=None):
    # Move to CPU and convert to numpy array
    x13_np = x13.cpu().numpy()
    x7_np = x7.cpu().numpy()
    x3_np = x3.cpu().numpy()
    
    # Reshape
    x13_np = x13_np[:, 0, :]  # [num_particles, D1]
    x7_np = x7_np[:, 0, :]  # [num_particles, D2]
    x3_np = x3_np[:, 0, :]  # [num_particles, D3]
    
    # Sampling (to reduce processing load)
    sample_size_13 = min(5000, x13_np.shape[0])
    sample_size_7 = min(5000, x7_np.shape[0])
    sample_size_3 = min(5000, x3_np.shape[0])
    sample_indices_13 = np.random.choice(x13_np.shape[0], sample_size_13, replace=False)
    sample_indices_7 = np.random.choice(x7_np.shape[0], sample_size_7, replace=False)
    sample_indices_3 = np.random.choice(x3_np.shape[0], sample_size_3, replace=False)
    x13_sampled = x13_np[sample_indices_13, :]
    x7_sampled = x7_np[sample_indices_7, :]
    x3_sampled = x3_np[sample_indices_3, :]

    print( x13_sampled.shape, x7_sampled.shape, x3_sampled.shape)

    reducer_13, reducer_7, reducer_3 = fixed_reducers
    # Use only transform (do not fit)
    x13_reduced = reducer_13.transform(x13_sampled)
    x7_reduced = reducer_7.transform(x7_sampled)
    x3_reduced = reducer_3.transform(x3_sampled)
    new_reducers = fixed_reducers  # Return the same reducers
    new_axis_limits = axis_limits

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(24, 8))
    
    # Plot x1
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(x13_reduced[:, 0], x13_reduced[:, 1], x13_reduced[:, 2], 
               c='b', marker='o', s=10, alpha=0.6)
    
    x13_limits = new_axis_limits[0]
    ax1.set_xlim(x13_limits[0], x13_limits[1])
    ax1.set_ylim(x13_limits[2], x13_limits[3])
    ax1.set_zlim(x13_limits[4], x13_limits[5])
    
    # Plot x2
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(x7_reduced[:, 0], x7_reduced[:, 1], x7_reduced[:, 2], 
               c='r', marker='o', s=10, alpha=0.6)
    
    x7_limits = new_axis_limits[1]
    ax2.set_xlim(x7_limits[0], x7_limits[1])
    ax2.set_ylim(x7_limits[2], x7_limits[3])
    ax2.set_zlim(x7_limits[4], x7_limits[5])
    
    # Plot x3
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(x3_reduced[:, 0], x3_reduced[:, 1], x3_reduced[:, 2], 
               c='g', marker='o', s=10, alpha=0.6)
    
    x3_limits = new_axis_limits[2]
    ax3.set_xlim(x3_limits[0], x3_limits[1])
    ax3.set_ylim(x3_limits[2], x3_limits[3])
    ax3.set_zlim(x3_limits[4], x3_limits[5])
    
    # Save
    plt.savefig(os.path.join(save_dir, f'triple_particles_iter_{iteration:04d}.png'))
    plt.close(fig)
    
    return new_reducers, new_axis_limits

def main():
    parser = argparse.ArgumentParser(description='Regenerate graphs using saved particle data')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory where particle data is saved')
    parser.add_argument('--reference_step', type=int, required=True, help='Reference step to determine axes')
    parser.add_argument('--output_dir', type=str, default='replay_graphs', help='Directory to save output graphs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()
    
    # Fix seed for reproducibility
    np.random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save command line arguments to a text file
    command_line_file = os.path.join(args.output_dir, 'command_line.txt')
    with open(command_line_file, 'w') as f:
        f.write(' '.join(sys.argv) + '\n')
    print(f"Command line saved to {command_line_file}")
    
    # Create path to data directory
    data_dir = Path(args.data_dir)
    
    # Get all pkl files
    pkl_files = sorted(glob.glob(str(data_dir / "particles_iter_*.pkl")))
    
    if not pkl_files:
        print(f"Error: No pkl files found in {data_dir}.")
        return
    
    # Extract iteration numbers from filenames and sort
    pkl_files_with_iter = [(f, extract_iteration_from_filename(f)) for f in pkl_files]
    pkl_files_with_iter.sort(key=lambda x: x[1])
    
    # Find file for reference step
    reference_file = None
    for file, iter_num in pkl_files_with_iter:
        if iter_num == args.reference_step:
            reference_file = file
            break
    
    if reference_file is None:
        print(f"Error: File for reference step {args.reference_step} not found.")
        return
    
    print(f"Reference file: {reference_file}")
    
    # Load reference file
    with open(reference_file, 'rb') as f:
        reference_data = pickle.load(f)
    
    # Get PCA information from reference data
    reference_x13 = reference_data.get('x13')
    reference_x7 = reference_data.get('x7')
    reference_x3 = reference_data.get('x3')
    
    # Move to CPU and convert to numpy array
    reference_x13_np = reference_x13.cpu().numpy()[:, 0, :]
    reference_x7_np = reference_x7.cpu().numpy()[:, 0, :]
    reference_x3_np = reference_x3.cpu().numpy()[:, 0, :]
    
    # Sampling
    sample_size = min(5000, reference_x13_np.shape[0])
    sample_indices = np.random.choice(reference_x13_np.shape[0], sample_size, replace=False)
    reference_x13_sampled = reference_x13_np[sample_indices, :]
    reference_x7_sampled = reference_x7_np[sample_indices, :]
    reference_x3_sampled = reference_x3_np[sample_indices, :]
    
    # Calculate PCA
    reducer_13 = PCA(n_components=3, random_state=args.seed)
    reducer_7 = PCA(n_components=3, random_state=args.seed)
    reducer_3 = PCA(n_components=3, random_state=args.seed)  # 3-dimensional data is used directly
        
    # Train PCA on reference data
    x13_reduced = reducer_13.fit_transform(reference_x13_sampled)
    x7_reduced = reducer_7.fit_transform(reference_x7_sampled)
    x3_reduced = reducer_3.fit_transform(reference_x3_sampled)
        
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
        np.min(x3_reduced[:, 0]) - 0.4, np.max(x3_reduced[:, 0]) + 0.4,
        np.min(x3_reduced[:, 1]) - 0.4, np.max(x3_reduced[:, 1]) + 0.4,
        np.min(x3_reduced[:, 2]) - 0.4, np.max(x3_reduced[:, 2]) + 0.4
    ]
        
    reducers = [reducer_13, reducer_7, reducer_3]
    axis_limits = [x13_limits, x7_limits, x3_limits]

    # Save reducers and axis_limits for reuse
    reducers_file = os.path.join(args.output_dir, 'reducers_and_axis_limits.pkl')
    with open(reducers_file, 'wb') as f:
        pickle.dump({'reducers': reducers, 'axis_limits': axis_limits}, f)
    print(f"Reducers and axis_limits saved to {reducers_file}")

    # Visualize data for each step
    for file, iter_num in pkl_files_with_iter:
        print(f"Processing: {file}, Iteration: {iter_num}")
        
        # Load file
        with open(file, 'rb') as f:
            data = pickle.load(f)
        
        # Get data
        x13 = data.get('x13')
        x7 = data.get('x7')
        x3 = data.get('x3')
        
        # Visualize using fixed axes
        visualize_particles_triple_fixed_axes(
            x13, x7, x3, iter_num, args.output_dir, 
            fixed_reducers=reducers,
            axis_limits=axis_limits
        )
    
    print(f"All graphs saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
