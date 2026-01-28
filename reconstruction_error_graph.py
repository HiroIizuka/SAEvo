import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Create reconstruction error graph up to a specified max iteration')
    parser.add_argument('--json_file', type=str, required=True, help='Path to reconstruction_errors.json file')
    parser.add_argument('--max_iter', type=int, required=True, help='Maximum iteration to plot')
    parser.add_argument('--output_file', type=str, default=None, help='Output file path (default: reconstruction_errors_max{max_iter}.png)')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory (default: current directory)')

    args = parser.parse_args()
    
    # Check if JSON file exists
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return
    
    # Load reconstruction error data
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    # Extract iterations and errors
    iterations = data.get('iterations', [])
    errors = data.get('errors', {})
    
    if not iterations:
        print("Error: No iteration data found in JSON file.")
        return
    
    if not errors:
        print("Error: No error data found in JSON file.")
        return
    
    # Filter data up to max_iter
    filtered_indices = [i for i, iter_num in enumerate(iterations) if iter_num <= args.max_iter]
    
    if not filtered_indices:
        print(f"Error: No data found up to iteration {args.max_iter}.")
        return
    
    filtered_iterations = [iterations[i] for i in filtered_indices]
    filtered_errors = {}
    for name, values in errors.items():
        filtered_errors[name] = [values[i] for i in filtered_indices]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output file path
    if args.output_file is None:
        output_file = os.path.join(args.output_dir, f'reconstruction_errors_max{args.max_iter}.png')
    else:
        output_file = args.output_file
        if not os.path.isabs(output_file):
            output_file = os.path.join(args.output_dir, output_file)
    
    # Create graph
    plt.figure(figsize=(10, 6))
    for name, values in filtered_errors.items():
        plt.plot(filtered_iterations, values, label=name, marker='o', markersize=3)
    
    plt.xlabel('Step', fontsize=16)
    plt.ylabel('Reconstruction Error', fontsize=16)
    #plt.title(f'Reconstruction Errors over Iterations')
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tick_params(labelsize=15)
    
    # Save graph
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graph saved to {output_file}")
    
    # Save command line arguments
    command_line_file = os.path.join(args.output_dir, 'command_line_reconstruction_error.txt')
    with open(command_line_file, 'w') as f:
        f.write(' '.join(sys.argv) + '\n')
    print(f"Command line saved to {command_line_file}")

if __name__ == "__main__":
    main()

