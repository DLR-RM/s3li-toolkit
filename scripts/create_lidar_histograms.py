import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from progressbar import ProgressBar
import pandas as pd


def create_progressbar():
    return ProgressBar(maxval=100)


def process_lidar_histograms(sequences, base_path, output_folder="histograms"):
    """
    Process LiDAR data from pickle files and create histogram images.

    Args:
        sequences (list): List of sequence names to process
        base_path (str): Path to base S3LI folder
        output_folder (str): Name of the folder to store histogram images
    """
    print("Processing LiDAR histograms for sequences:", sequences)

    for seq_name in sequences:
        print(f"\nProcessing sequence: {seq_name}")

        # Setup paths
        dataset_path = os.path.join(base_path, "dataset", seq_name)
        hist_dst = os.path.join(dataset_path, output_folder)
        pickle_path = os.path.join(dataset_path, f"{seq_name}.pkl")

        # Create output directory if it doesn't exist
        if not os.path.exists(hist_dst):
            os.makedirs(hist_dst)

        print("Output folder:", hist_dst)
        print("Reading Pickle from:", pickle_path)

        # Read pickle file
        df = pd.read_pickle(pickle_path)

        # Initialize histogram path column
        df['histogram_path'] = None

        print(f"Processing {len(df)} entries...")

        # Setup progress bar
        progress = create_progressbar()
        progress.start()

        # Process each entry
        for idx, row in df.iterrows():
            # Update progress
            progress.update((idx / len(df)) * 100)

            # Get point cloud directly from pickle (no need to parse)
            points = row['point_cloud']
            time_stamp = row['time_stamp']

            # Extract Z coordinates (depths)
            depths = points[:, 2]

            # Create histogram
            plt.figure(figsize=(8, 6))
            plt.hist(depths, bins=50, color='navy', alpha=0.7)
            plt.title(f'LiDAR Depth Distribution\nTimestamp: {time_stamp:.2f}')
            plt.xlabel('Depth (m)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)

            # Add statistics to the plot
            stats_text = f'Mean: {np.mean(depths):.2f}m\n'
            stats_text += f'Median: {np.median(depths):.2f}m\n'
            stats_text += f'Std: {np.std(depths):.2f}m\n'
            stats_text += f'Points: {len(depths)}'
            plt.text(0.95, 0.95, stats_text,
                     transform=plt.gca().transAxes,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Save histogram
            hist_path = os.path.join(hist_dst, f'hist_{time_stamp:.6f}.png')
            plt.savefig(hist_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Store histogram path in DataFrame
            df.at[idx, 'histogram_path'] = hist_path

        progress.finish()

        # Save updated DataFrame back to pickle
        df.to_pickle(pickle_path)
        print(f"Updated pickle file with histogram paths: {pickle_path}")

        print(f"Finished processing {seq_name}")
        print(f"Processed timestamps from {df['time_stamp'].min():.2f} to {df['time_stamp'].max():.2f}")
        print(f"Histograms saved to: {hist_dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create LiDAR histogram images from S3LI dataset pickle files')
    parser.add_argument('path', type=str, help='path to base S3LI folder')
    parser.add_argument('--output', type=str, default='histograms',
                        help='name of the output folder for histogram images')

    args = parser.parse_args()

    sequences = ["s3li_traverse_2",
                 "s3li_loops",
                 "s3li_traverse_1",
                 "s3li_crater",
                 "s3li_crater_inout",
                 "s3li_mapping",
                 "s3li_landmarks"]

    process_lidar_histograms(sequences, args.path, args.output)
