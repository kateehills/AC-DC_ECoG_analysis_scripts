#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to process and analyze ECoG data without Z-scoring.
Exports normalized PSD and overall power/SEM.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

# %% Load Data from .h5 File
def load_data(file_path, group_path):
    """Load 1 hour of ECoG data from an .h5 file."""
    with h5py.File(file_path, 'r') as h5file:
        data = h5file[group_path]['data'][:]
    return data

# %% Compute FFT
def compute_fft(data, sample_rate=256):
    """Compute FFT for data segment and return frequencies and magnitudes."""
    fft_values = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1/sample_rate)
    return freqs, fft_values

# %% Compute Power Spectral Density (PSD) per Second
def compute_psd_per_second(data, sample_rate=256, bands=None):
    """Compute PSD per second for given frequency bands."""
    if bands is None:
        bands = {
            'Delta (1-4 Hz)': (1, 4), 
            'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-12 Hz)': (8, 12), 
            'Beta (12-30 Hz)': (12, 30),
            'Gamma (30-80 Hz)': (30, 80)
        }

    num_seconds = len(data) // sample_rate
    psd_values = {band: [] for band in bands}

    for sec in range(num_seconds):
        start = sec * sample_rate
        end = start + sample_rate
        segment = data[start:end]
        
        # Compute FFT for this 1s segment
        freqs, fft_values = compute_fft(segment, sample_rate)

        for band, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.sum(fft_values[idx]**2) / (high - low)
            psd_values[band].append(band_power)

    return psd_values

# %% Compute Mean Baseline PSD (First 200s)
def compute_baseline_mean_psd(psd_values, baseline_duration=200):
    """Compute mean baseline PSD for each frequency band over the first 200s."""
    mean_baseline_psd = {band: np.mean(psd_values[band][:baseline_duration]) for band in psd_values}
    return mean_baseline_psd

# %% Normalize PSD Values Relative to Baseline
def normalize_psd(psd_values, mean_baseline_psd):
    """Normalize PSD values relative to the mean baseline PSD per frequency band."""
    normalized_psd = {band: [x / mean_baseline_psd[band] for x in psd_values[band]] for band in psd_values}
    return normalized_psd

# %% Calculate overall power
def calculate_overall_power_by_band(normalized_psd_values):
    """
    Computes overall power by first averaging within each frequency band.
    Then averages these band-level values for an overall measure.
    """
    band_means = []
    
    for band, values in normalized_psd_values.items():
        band_means.append(np.mean(values))  # Compute mean per band

    # Compute overall mean and SEM from band-level means
    overall_mean_power = np.mean(band_means)  
    overall_sem = np.std(band_means) / np.sqrt(len(band_means))

    return overall_mean_power, overall_sem

# %% Main Script
if __name__ == "__main__":
    # File details
    h5_filepath = '/Volumes/KH_PhD_01/P1/GRE_seizure_files/Seizure_h5/2203_TID213/M1647455392_2022-03-16-18-29-52_tids_[213].h5'
    hierarchy = 'M1647455392/213'

    # Load data (1 hour of ECoG)
    data = load_data(h5_filepath, hierarchy)
    sample_rate = 256

    # Compute PSD per second
    psd_values = compute_psd_per_second(data, sample_rate)

    # Compute baseline mean PSD (first 200s)
    mean_baseline_psd = compute_baseline_mean_psd(psd_values)

    # Normalize PSD values
    normalized_psd = normalize_psd(psd_values, mean_baseline_psd)

    # Compute overall power for full 1-hour dataset
    overall_power_hour, overall_sem_hour = calculate_overall_power_by_band(normalized_psd)

    # Convert normalized PSD to DataFrame
    df_norm_psd = pd.DataFrame(normalized_psd)

# %% Select Seizure Period and Compute PSD
    seizure_start = 1800 #600 #600 #800 #1080 #350 #530  
    seizure_end = 1950 #750 #800 #900 #1180 #450 #630

    seizure_start_idx = int(seizure_start * sample_rate)
    seizure_end_idx = int(seizure_end * sample_rate)

    seizure_data = data[seizure_start_idx:seizure_end_idx]

    seizure_psd_values = compute_psd_per_second(seizure_data, sample_rate)

    seizure_normalized_psd = normalize_psd(seizure_psd_values, mean_baseline_psd)

    overall_power_seizure, overall_sem_seizure = calculate_overall_power_by_band(seizure_normalized_psd)

    df_seizure_norm_psd = pd.DataFrame(seizure_normalized_psd)

# %% Export to Excel
tid = hierarchy.split('/')[-1]  
h5_filename = h5_filepath.split('/')[-1]  
formatted_hierarchy = hierarchy.replace("/", "_")  
excel_filename = f"normalized_psd_{formatted_hierarchy}.xlsx"

def format_dataframe_for_export(df, total_seconds):
    """Formats DataFrame to include TID, .h5 File Name, and Time Column."""
    formatted_df = df.copy()
    formatted_df.insert(0, "Second", np.arange(1, total_seconds + 1))
    formatted_df.insert(0, ".h5 File", h5_filename)
    formatted_df.insert(0, "TID", tid)
    return formatted_df

df_norm_psd_export = format_dataframe_for_export(df_norm_psd, total_seconds=3600)
df_seizure_norm_psd_export = format_dataframe_for_export(df_seizure_norm_psd, total_seconds=len(df_seizure_norm_psd))

def insert_overall_power(df, overall_power, overall_sem):
    df = df.copy()
    gamma_index = df.columns.get_loc("Gamma (30-80 Hz)")
    df.insert(gamma_index + 1, "Overall Power (µV²/Hz)", [overall_power] + [np.nan] * (len(df) - 1))
    df.insert(gamma_index + 2, "Overall SEM", [overall_sem] + [np.nan] * (len(df) - 1))
    return df

df_norm_psd_export = insert_overall_power(df_norm_psd_export, overall_power_hour, overall_sem_hour)
df_seizure_norm_psd_export = insert_overall_power(df_seizure_norm_psd_export, overall_power_seizure, overall_sem_seizure)

with pd.ExcelWriter(excel_filename) as writer:
    df_norm_psd_export.to_excel(writer, sheet_name="Full Hour PSD", index=False)
    df_seizure_norm_psd_export.to_excel(writer, sheet_name="Seizure PSD", index=False)

print(f"Exported: {excel_filename}")

# %% Plot Full 1-Hour ECoG Time Series
plt.figure(figsize=(12, 5))
time_axis = np.arange(len(data)) / sample_rate  # Convert samples to time in seconds
plt.plot(time_axis, data, color="black", linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title("ECoG Time Series (1 Hour)")
plt.xlim(0, len(data) / sample_rate)

# Remove top and right spines, keeping only x and y axes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.75)  # Keep y-axis frame
ax.spines['bottom'].set_linewidth(0.75)  # Keep x-axis frame

plt.savefig(f"ECoG_FullHour_{h5_filename}_TID{tid}.png", dpi=300, bbox_inches='tight', transparent=True)

# %% Plot Seizure Window ECoG
plt.figure(figsize=(12, 5))
seizure_time_axis = np.arange(len(seizure_data)) / sample_rate + seizure_start
plt.plot(seizure_time_axis, seizure_data, color="green", linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title(f"ECoG Time Series (Seizure Window: {seizure_start}s - {seizure_end}s)")
plt.xlim(seizure_start, seizure_end)

# Remove top and right spines, keeping only x and y axes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.75)  # Keep y-axis frame
ax.spines['bottom'].set_linewidth(0.75)  # Keep x-axis frame
plt.savefig(f"ECoG_Seizure_{h5_filename}_TID{tid}.png", dpi=300, bbox_inches='tight', transparent=True)

