#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated script for plotting ECoG, FFT, and PSD from a single .h5 file.
Now includes:
- Correct PSD calculation in 1-second increments (total of 3600 PSD values for 1 hour)
- Mean and SEM displayed with error bars for each frequency band
- Pop-out, interactive zoom, and subplot saving functionality
- Option to normalize PSD values to a baseline (first 300 seconds)
- Addition of normalized percentage of baseline
- Plot of raw, normalized, and percentage baseline PSD in a separate figure
- Handles empty data during zoom
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import zscore
import pandas as pd
import sys

plt.close('all')

# %% Load Data from .h5 File
def load_data_from_h5(file_path, group_path):
    """Load numerical time-series data from an HDF5 file, structured identically to baseline data loading."""
    dataset_path = f"{group_path}/data"  # Ensure consistent dataset path format

    try:
        with h5py.File(file_path, 'r') as h5file:
            if dataset_path in h5file:  # Check if dataset exists
                analysis_data = h5file[dataset_path][()]  # Load dataset identically to baseline
                print(f"✅ Successfully loaded analysis time-series data from {file_path} at {dataset_path}.")
                return analysis_data
            else:
                print(f"❌ Dataset {dataset_path} not found in {file_path}. Skipping...")
                return None
    except Exception as e:
        print(f"❌ Error loading analysis time-series data: {e}. Skipping...")
        return None

# %% Compute FFT and Return Frequencies and FFT Values
def compute_fft(data, sample_rate=256):
    """Compute FFT and return frequencies and FFT values (magnitude).
    - If the input data is empty, return empty arrays.
    """
    if len(data) == 0:  # Handle zero-length data
        return np.array([]), np.array([])
    fft_values = np.abs(np.fft.rfft(data))  # FFT magnitude
    freqs = np.fft.rfftfreq(len(data), 1/sample_rate)  # FFT frequency axis

    return freqs, fft_values

# %% Load Baseline PSD for 1h5
def load_baseline_data(file_path, hierarchy):
    """
    Load raw time-series baseline data from an external .h5 file.
    """
    try:
        with h5py.File(file_path, 'r') as h5_file:
            dataset_path = f"{hierarchy}/data"
            
            if dataset_path in h5_file:
                baseline_data = h5_file[dataset_path][()]
                print(f"✅ Successfully loaded baseline time-series data from {file_path} at {dataset_path}.")
                return baseline_data
            else:
                print(f"❌ Dataset {dataset_path} not found in {file_path}. Falling back to in-script calculation.")
                return None
    except Exception as e:
        print(f"❌ Error loading baseline time-series data: {e}. Falling back to in-script calculation.")
        return None
    
# %% Compute Baseline PSD for 1h5
def compute_baseline_psd(data, sample_rate=256, window_size=1, bands=None):
    """
    Compute the baseline PSD for the first 200 seconds of the dataset.
    The PSD is calculated for each 1-second window, and the mean PSD across 200 windows is returned.
    """
    if data is None:
        print("❌ No valid baseline data available. Cannot compute baseline PSD.")
        return None, None

    if bands is None:
        bands = {'Delta (1-4 Hz)': (1, 4), 'Theta (4-8 Hz)': (4, 8),
                 'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
                 'Gamma (30-80 Hz)': (30, 80)}

    num_samples_per_window = window_size * sample_rate
    total_baseline_windows = 200

    psd_values = {band: [] for band in bands}

    for window_idx in range(total_baseline_windows):
        start = window_idx * num_samples_per_window
        end = start + num_samples_per_window
        window_data = data[start:end]

        # Compute FFT
        freqs, fft_values = compute_fft(window_data, sample_rate)

        for band, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_psd = np.sum(fft_values[idx]**2) / (high - low)
            psd_values[band].append(band_psd)

    # Compute the mean PSD for each band
    mean_raw_baseline_psd = {band: np.mean(psd_values[band]) for band in bands}
    print("✅ Baseline PSD computed from first 200 seconds of time-series data.")

    # Convert per-second PSD values into a DataFrame for review
    df_baseline_psd = pd.DataFrame(psd_values)
    df_baseline_psd.insert(0, "Time (s)", np.arange(1, total_baseline_windows + 1))

    print("✅ Baseline PSD DataFrame created for review.")
    return mean_raw_baseline_psd, df_baseline_psd

#%% Calculate the band powers per frequency band in 1 second windows, normalise them to the 200s baseline period 
def compute_band_powers_in_windows(data, sample_rate=256, window_size=1, bands=None, z_threshold=3, mean_raw_baseline_psd=None):
    """Compute band powers and normalized PSD in 1-second windows over 1 hour."""
    
    if data is None:
        print("Warning: No valid analysis data available. Skipping analysis.")
        return None, None, None, None, None, None, None, None, None
    
    if bands is None:
        bands = {
            'Delta (1-4 Hz)': (1, 4), 'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
            'Gamma (30-80 Hz)': (30, 80)
        }

    num_samples_per_window = window_size * sample_rate
    total_windows = len(data) // num_samples_per_window

    psd_values = {band: [] for band in bands}

    # Compute PSD values for each window
    for window_idx in range(total_windows):
        start = window_idx * num_samples_per_window
        end = start + num_samples_per_window
        window_data = data[start:end]
        freqs, fft_values = compute_fft(window_data, sample_rate)

        for band, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_psd = np.sum(fft_values[idx]**2) / (high - low)
            psd_values[band].append(band_psd)

    # Normalize PSD values if baseline is available
    if not mean_raw_baseline_psd:
        mean_raw_baseline_psd = {band: np.mean(psd_values[band][:200]) for band in bands}
        print("Baseline PSD (used for normalization):", mean_raw_baseline_psd)

    normalized_psd_values = {
        band: [x / mean_raw_baseline_psd[band] for x in psd_values[band]] for band in bands
    }

    # Apply Z-score filtering
    filtered_psd, mean_filtered_psd = apply_zscore_per_band(normalized_psd_values, z_threshold, mean_raw_baseline_psd)

    # Compute additional outputs
    percent_psd_values = {band: [x * 100 for x in filtered_psd[band]] for band in bands}
    band_powers = {
        band: {
            'mean': np.nanmean(filtered_psd[band]),
            'sem': np.nanstd(filtered_psd[band]) / np.sqrt(np.count_nonzero(~np.isnan(filtered_psd[band])))
        }
        for band in bands
    }
    normalized_band_powers = {band: np.nanmean(normalized_psd_values[band]) for band in bands}
    overall_mean_power = np.nanmean([band_powers[band]['mean'] for band in bands if not np.isnan(band_powers[band]['mean'])])
    overall_sem = np.nanmean([band_powers[band]['sem'] for band in bands if not np.isnan(band_powers[band]['sem'])])

    # ✅ Correctly return 9 values
    return (
        band_powers, psd_values, normalized_psd_values, percent_psd_values,
        filtered_psd, mean_filtered_psd, normalized_band_powers, overall_mean_power, overall_sem
    )

#%% Apply z scoring per frequency band over 1 hour period 
def apply_zscore_per_band(normalized_psd, z_threshold=3, baseline_psd=None, label='3600s'):
    """
    Apply Z-scoring within each frequency band and preserve NaN placement.
    """
    filtered_psd = {}
    mean_filtered_psd = {}
    
    print(f"\nZ-Score Thresholds for Outlier Removal ({label} Dataset):")
    print("-" * 50)  # Decorative separator
    
    for band, values in normalized_psd.items():
        mean = np.mean(values)
        std_dev = np.std(values)
        min_threshold = mean - (z_threshold * std_dev)
        max_threshold = mean + (z_threshold * std_dev)
        
        # Print each band's thresholds in an aligned format
        print(f"{band:<18} Z-score thresholds: Min = {min_threshold:>7.4f}, Max = {max_threshold:>7.4f}")

        # Preserve original length, replacing outliers with NaN
        filtered_psd[band] = np.array([x if min_threshold <= x <= max_threshold else np.nan for x in values])

        # Compute mean baseline PSD per band, ignoring NaNs
        mean_filtered_psd[band] = np.nanmean(values) 

    print("-" * 50)  # Decorative separator
    
    return filtered_psd, mean_filtered_psd

#%% Calculate mean and sem per frequency band over 1 hour period 
def calculate_mean_sem_per_band(filtered_psd):
    """
    Calculate mean and SEM individually for each frequency band and return as a DataFrame.
    
    Args:
        filtered_psd (dict): Dictionary containing filtered PSD values (with NaNs).
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Frequency Band', 'Mean', 'SEM'].
    """
    mean_values = []
    sem_values = []
    bands = []

    for band, values in filtered_psd.items():
        valid_values = np.array(values)[~np.isnan(values)]  # Remove NaNs

        if len(valid_values) > 0:
            mean_value = np.nanmean(valid_values)  # Mean ignoring NaNs
            sem_value = np.nanstd(valid_values) / np.sqrt(len(valid_values))  # SEM ignoring NaNs
        else:
            mean_value = np.nan  # If all values are NaN, return NaN
            sem_value = np.nan

        bands.append(band)
        mean_values.append(mean_value)
        sem_values.append(sem_value)

    # Create a DataFrame
    df_mean_sem = pd.DataFrame({'Frequency Band': bands, 'Mean': mean_values, 'SEM': sem_values})
    
    return df_mean_sem

#%% Conversion of filtered_psd and mean/sem within frequency bands to dataframe
def convert_filtered_psd_to_dataframe(filtered_psd):
    """
    Convert filtered normalized PSD values into a DataFrame for export.

    Args:
        filtered_psd (dict): Dictionary containing filtered PSD values (with NaNs).

    Returns:
        pd.DataFrame: DataFrame with columns ['Time (s)', 'Frequency Band', 'Filtered PSD'].
    """
    data = []

    for band, values in filtered_psd.items():
        for time_idx, value in enumerate(values):
            data.append({'Time (s)': time_idx, 'Frequency Band': band, 'Filtered PSD': value})

    # Convert list to DataFrame
    df_filtered_psd = pd.DataFrame(data)

    return df_filtered_psd

# %% Plot Time Series (ECoG trace)
def plot_time_series(ax, data, sample_rate=256, color='k', linewidth=0.5, label=None):
    """Plot the time series ECoG data on a subplot."""
    time = np.arange(len(data)) / sample_rate  # Time axis
    total_time = len(data) / sample_rate  # Total duration of the recording in seconds

    # Ensure time axis is within the recording length (0 to total_time)
    ax.plot(time, data, color=color, linewidth=linewidth, label=label)
    ax.set_xlim(0, total_time)  # Clamp x-axis to valid time range
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if label:
        ax.legend()

# %% Plot FFT (Frequency Power Spectrum)
def plot_fft(ax, freqs, fft_values, color='b', label=None):
    """Plot the Fast Fourier Transform (FFT) on the given axis."""
    ax.plot(freqs, fft_values, color=color, linewidth=0.5, label=label)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(0, 100)
    ax.spines['top'].set_visible(False)  # Remove top frame (spine)
    ax.spines['right'].set_visible(False)  # Remove right frame (spine)
    if label:
        ax.legend()

# %% Plot Band Powers as Line Plot with SEM
def plot_band_powers(ax, band_powers, label=None):
    """
    Plot power spectral density in specific frequency bands as a line plot with SEM error bars.
    This plot uses the normalized PSD values.
    """
    bands = list(band_powers.keys())  # Frequency band labels
    means = [band_powers[band]['mean'] for band in bands]  # Mean values
    sems = [band_powers[band]['sem'] for band in bands]  # SEM values

    # Set evenly spaced x-axis values
    x_values = np.arange(len(bands))  # Evenly spaced integers for each band

    # Plot the line with error bars (SEM)
    ax.errorbar(x_values, means, yerr=sems, fmt='-o', color='c', label=label)

    # Set the labels and axis limits
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Normalized Power')
    ax.set_xticks(x_values)  # Use evenly spaced x-axis values
    ax.set_xticklabels(bands, rotation=45, ha="right")  # Display the band names

    if label:
        ax.legend()

    # Set background to transparent
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
# %% Export PSD to excel with overall mean power and SEM
def export_psd_to_excel(filtered_normalized_psd_values, file_path, overall_mean_power, overall_sem):
    """
    Export PSD values to an Excel file for each processed .h5 file.

    Args:
        filtered_normalized_psd_values (dict): Dictionary containing filtered normalized PSD values.
        file_path (str): Path of the current analysis .h5 file being processed.
        overall_mean_power (float): Overall mean power across all bands.
        overall_sem (float): Overall SEM across all bands.

    Returns:
        str: Path to the exported Excel file.
    """

    # Extract filename from file path
    h5_filename = os.path.basename(file_path)
    
    # Extract the dynamic hierarchy (TID) from filename
    tid = os.path.splitext(h5_filename)[0]  # Removes .h5 extension

    # Handle empty dictionary case
    if not filtered_normalized_psd_values:
        print(f"⚠ Warning: No PSD values to export for {h5_filename}. Skipping Excel export.")
        return None

    # Find the maximum length of any band to preserve NaN placements
    max_len = max(len(values) for values in filtered_normalized_psd_values.values())

    # Ensure NaN gaps are preserved by initializing full NaN arrays
    padded_data = {band: np.full(max_len, np.nan) for band in filtered_normalized_psd_values}

    # Fill in original values while keeping NaN placements intact
    for band, values in filtered_normalized_psd_values.items():
        padded_data[band][:len(values)] = values  # Maintain NaN placements

    # Convert to DataFrame
    df = pd.DataFrame(padded_data)

    # Add metadata columns
    df.insert(0, 'TID', tid)
    df.insert(1, '.h5 File', h5_filename)
    df.insert(2, 'Time Index', range(1, max_len + 1))

    # Add overall mean power and SEM, keeping only the first row filled
    df['Overall Mean Power'] = [overall_mean_power] + [np.nan] * (max_len - 1)
    df['Overall SEM'] = [overall_sem] + [np.nan] * (max_len - 1)

    # Define column order explicitly
    column_order = ['TID', '.h5 File', 'Time Index'] + list(filtered_normalized_psd_values.keys()) + ['Overall Mean Power', 'Overall SEM']
    df = df[column_order]

    # Define the export path using the same folder as the input file
    excel_file_name = h5_filename.replace('.h5', '_filtered_normalized_psd.xlsx')
    excel_file_path = os.path.join(os.path.dirname(file_path), excel_file_name)

    # Export to Excel
    df.to_excel(excel_file_path, index=False)
    print(f"✅ Data exported successfully to {excel_file_path}")

    return excel_file_path

# %% Main Script (Master Plot)
if __name__ == "__main__":
    # Hardcoded folder path (replace with your actual path)
    FOLDER_PATH = '/Users/katehills/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Thesis/GRE_characterisation/Ephys/Baseline_files-exports/Controls/2304_58'

    # Check if the folder exists
    if not os.path.isdir(FOLDER_PATH):
        print("❌ Invalid folder path. Exiting...")
        sys.exit()

    print(f"🔍 Looking for .h5 files in: {FOLDER_PATH}")

    # List .h5 files in the directory
    h5_files = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) if f.endswith(".h5")]

    # Check if any .h5 files exist
    if not h5_files:
        print("❌ No .h5 files found in the folder. Listing folder contents:")
        print(os.listdir(FOLDER_PATH))
        sys.exit()

    print(f"✅ Found {len(h5_files)} .h5 files.")

    # Load baseline data once
    baseline_h5_filepath = '/Users/katehills/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Thesis/GRE_characterisation/Ephys/Baseline_files-exports/Controls/2304_58/M1680844577_2023-04-07-06-16-17_tids_[58].h5'
    baseline_hierarchy = "M1680844577/58"
    baseline_data = load_baseline_data(baseline_h5_filepath, baseline_hierarchy)
    mean_raw_baseline_psd, _ = compute_baseline_psd(baseline_data, sample_rate=256)

    # Process each .h5 file in the folder
    for analysis_h5_filepath in h5_files:
        print(f"\n🔄 Processing file: {analysis_h5_filepath}")

        # Extract root group dynamically
        with h5py.File(analysis_h5_filepath, 'r') as h5file:
            root_keys = list(h5file.keys())  # Get top-level groups
            print(f"🔍 Found root groups in {analysis_h5_filepath}: {root_keys}")

            if not root_keys:
                print(f"❌ No valid group found in {analysis_h5_filepath}. Skipping.")
                continue

            root_group = root_keys[0]  # Assuming a single root group
            analysis_hierarchy = f"{root_group}/58"  # Do NOT append "/data" yet

        # Print final dataset path before loading
        print(f"📂 Using dataset path: {analysis_hierarchy}/data")

        # Load data using the modified function
        analysis_data = load_data_from_h5(analysis_h5_filepath, analysis_hierarchy)
        
        if analysis_data is None:
            print(f"❌ Skipping {analysis_h5_filepath} - No valid data found.")
            continue

        print(f"✅ Successfully loaded data for {analysis_h5_filepath}, shape: {analysis_data.shape}")

        # Compute PSD and band powers
        band_powers, psd_values, normalized_psd_values, percent_psd_values, filtered_normalized_psd_values, mean_filtered_psd, normalized_band_powers, overall_mean_power, overall_sem = compute_band_powers_in_windows(
            analysis_data, sample_rate=256, mean_raw_baseline_psd=mean_raw_baseline_psd
        )

        # Export PSD results
        export_psd_to_excel(filtered_normalized_psd_values, analysis_h5_filepath, overall_mean_power, overall_sem)

        print(f"✅ Successfully processed: {analysis_h5_filepath}")

        # -------------------------------
        # ✅ Move plotting INSIDE the loop
        # -------------------------------
        # Create figure and GridSpec to control width ratios (70% ECoG, 15% FFT, 15% Band Power)
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, width_ratios=[6.5, 1.75, 1.75], figure=fig)

        # Create subplots
        ax1 = fig.add_subplot(gs[0])  # ECoG trace (70% width)
        ax2 = fig.add_subplot(gs[1])  # FFT (15% width)
        ax3 = fig.add_subplot(gs[2])  # Band Power (15% width)

        # Plot the ECoG trace for the analysis data
        plot_time_series(ax1, analysis_data, sample_rate=256, color='purple', linewidth=1, label='ECoG')

        # Initial FFT for the analysis dataset
        freqs, fft_values = compute_fft(analysis_data)
        plot_fft(ax2, freqs, fft_values, color='purple', label='FFT')

        # Plot normalized band powers with SEM as a line plot
        plot_band_powers(ax3, band_powers, label='FNBP')

        # Print overall power and SEM
        print(f"\nOverall Mean Power (1-Hour): {overall_mean_power:.4f} ± {overall_sem:.4f} (SEM)")

        # Export analysis PSD to excel
        export_psd_to_excel(filtered_normalized_psd_values, analysis_h5_filepath, overall_mean_power, overall_sem)

        # Save the figure using the same naming convention as the Excel file
        h5_filename = os.path.basename(analysis_h5_filepath).replace('.h5', '_figure.png')
        figure_path = os.path.join(os.path.dirname(analysis_h5_filepath), h5_filename)
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved successfully at {figure_path}")

        # Show the interactive figure
        plt.tight_layout()
        plt.show()

    # Print completion message AFTER loop ends
    print("🎉 Processing complete for all files.")

