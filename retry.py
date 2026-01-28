import torch
import pickle
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from particle import Conv1DNet, ConvTranspose1DNet, update_particles, visualize_high_dim_particles_triple, visualize_high_dim_particles_triple_fixed_axes
from sklearn.decomposition import PCA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description='Run one simulation using saved particle data')
    #parser.add_argument('--data_dir', type=str, default='saved_particles', help='Directory where particle data is saved')
    parser.add_argument('--file', type=str, default=None, help='Specific pkl file name to use (if not specified, use the latest file)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--reducers_file', type=str, default=None, help='Path to pkl file containing reducers and axis_limits (e.g., from replay_forgraph.py output)')
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to save output graph (default: ./)')
    parser.add_argument('--output_file', type=str, default=None, help='Output file name (if not specified, uses default naming: triple_pca_particles_iter_0001.png)')

    args = parser.parse_args()
    
    # Fix seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {args.seed}")
    
    # Save command line arguments to a text file
    command_line_file = 'command_line_retry.txt'
    with open(command_line_file, 'w') as f:
        f.write(' '.join(sys.argv) + '\n')
    print(f"Command line saved to {command_line_file}")
    
    # If a specific file is specified
    if args.file:
        pkl_file = Path(args.file)  # Convert string to Path object
        if not pkl_file.exists():
            print(f"Error: Specified file {pkl_file} not found.")
            return
        print(f"Loading specified pkl file: {pkl_file}")
    else:
        # Search for the latest pkl file
        data_dir = Path('saved_particles')
        pkl_files = list(data_dir.glob('*.pkl'))
        if not pkl_files:
            print(f"Error: No pkl files found in {data_dir}.")
            return
        
        # Get the latest file (sorted by filename)
        pkl_file = sorted(pkl_files)[-1]
        print(f"Loading latest pkl file: {pkl_file}")
    
    # Load pkl file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get data
    x13 = data.get('x13')
    x7 = data.get('x7')
    x3 = data.get('x3')
    
    # Also get PCA information
    #reducers = data.get('reducers')
    #axis_limits = data.get('axis_limits')
    
    # Check device and move tensors to CPU if necessary
    device = torch.device("cpu")  # Perform all processing on CPU
    
    # Move tensors to CPU
    x13 = x13.to(device)
    x7 = x7.to(device)
    x3 = x3.to(device)

    noise_strength = 0.1
    
    ref_x13 = x13.cpu().numpy()[:, 0, :]
    ref_x7 = x7.cpu().numpy()[:, 0, :]
    ref_x3 = x3.cpu().numpy()[:, 0, :]

    num_particles = x13.shape[0]
    x13_length = x13.shape[2]
    x7_length = x7.shape[2]
    x3_length = x3.shape[2]

    model13_7 = Conv1DNet(output_channels=num_particles, kernel_size=x13_length, padding=3).to(device)
    model13_7.set_weight(x13)

    model7_13 = ConvTranspose1DNet(output_channels=num_particles, kernel_size=x7_length, padding=0).to(device)
    model7_13.set_weight(x7.permute(1, 0, 2))

    model7_3 = Conv1DNet(output_channels=num_particles, kernel_size=x7_length, padding=1).to(device)
    model7_3.set_weight(x7)

    model3_7 = ConvTranspose1DNet(output_channels=num_particles, kernel_size=x3_length, padding=0, stride=2).to(device)
    model3_7.set_weight(x3.permute(1, 0, 2))

    # Set models to evaluation mode
    model13_7.eval()
    model7_13.eval()
    model7_3.eval()
    model3_7.eval()
    
    activation = 'tanh'

    # Try to load reducers and axis_limits from file if specified
    reducers = None
    axis_limits = None
    if args.reducers_file:
        reducers_path = Path(args.reducers_file)
        if reducers_path.exists():
            print(f"Loading reducers and axis_limits from {reducers_path}")
            with open(reducers_path, 'rb') as f:
                reducers_data = pickle.load(f)
            reducers = reducers_data.get('reducers')
            axis_limits = reducers_data.get('axis_limits')
            if reducers and axis_limits:
                print("Successfully loaded reducers and axis_limits from file")
            else:
                print("Warning: reducers or axis_limits not found in file, will calculate new ones")
                reducers = None
                axis_limits = None
        else:
            print(f"Warning: reducers_file {reducers_path} not found, will calculate new ones")
    
    # Calculate reducers and axis_limits if not loaded from file
    if reducers is None or axis_limits is None:
        print("Calculating new reducers and axis_limits")
        reducer_13 = PCA(n_components=3, random_state=args.seed)
        reducer_7 = PCA(n_components=3, random_state=args.seed)
        reducer_3 = PCA(n_components=3, random_state=args.seed)  # 3-dimensional data is used directly
            
            # Train PCA on reference data
        x13_reduced = reducer_13.fit_transform(ref_x13)
        x7_reduced = reducer_7.fit_transform(ref_x7)
        x3_reduced = reducer_3.fit_transform(ref_x3)
            
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

    # Calculate reconstruction error
    with torch.no_grad():
            temp_out_3 = []
            temp_out_7 = []
            temp_out_13 = []

            out3_7 = model3_7(x3)
            out3_7 = out3_7.view(num_particles*num_particles, 1, x7_length)
            temp_out_7.append(out3_7)

            out7 = torch.cat(temp_out_7, dim=0)
            new_x7 = update_particles(x7, out7, 0.0, device, activation)
            new_x7 = new_x7 + torch.randn_like(x7) * noise_strength

            out7_13 = model7_13(new_x7)
            out7_13 = out7_13.view(num_particles*num_particles, 1, x13_length)
            temp_out_13.append(out7_13)

            out13 = torch.cat(temp_out_13, dim=0)
            new_x13 = update_particles(x13, out13, 0.0, device, activation)
            new_x13 = new_x13 + torch.randn_like(x13) * noise_strength




    print(new_x13.shape, new_x7.shape, x3.shape)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize using saved PCA information
    if reducers and axis_limits:
        visualize_high_dim_particles_triple_fixed_axes(
            new_x13, new_x7, x3, 1, str(output_dir), 
            method='pca', 
            fixed_reducers=reducers,
            fix_axes_after=0,
            axis_limits=axis_limits,
            random_state=args.seed
        )
    else:
        # If PCA information is not available, perform normal visualization
        visualize_high_dim_particles_triple(new_x13, new_x7, x3, 0, str(output_dir), method='pca', random_state=args.seed)
    
    # If custom output file name is specified, rename the default output file
    if args.output_file:
        default_filename = output_dir / 'triple_pca_particles_iter_0001.png'
        custom_filename = output_dir / args.output_file
        if default_filename.exists():
            default_filename.rename(custom_filename)
            print(f"Output saved to {custom_filename}")
        else:
            # Try alternative default name (for visualize_high_dim_particles_triple)
            default_filename_alt = output_dir / 'triple_pca_particles_iter_0000.png'
            if default_filename_alt.exists():
                default_filename_alt.rename(custom_filename)
                print(f"Output saved to {custom_filename}")
            else:
                print(f"Warning: Could not find default output file to rename. Expected {default_filename} or {default_filename_alt}")
    else:
        print(f"Output saved to {output_dir}")

if __name__ == "__main__":
    main()
