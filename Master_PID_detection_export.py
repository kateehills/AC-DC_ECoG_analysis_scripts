#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:32:08 2025

@author: katehills
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch, iirnotch, filtfilt
import os
from datetime import datetime

plt.close('all')

#====================================================================
# Configuration Constants
#====================================================================
SAMPLE_RATE = 256

DEFAULT_BANDS = {
    'Delta (1-4 Hz)': (1, 4),
    'Theta (4-8 Hz)': (4, 8),
    'Alpha (8-12 Hz)': (8, 12),
    'Beta (12-30 Hz)': (12, 30),
    'Gamma (30-80 Hz)': (30, 80)
}

# Notch filter parameters
NOTCH_FREQ = 50.0     # Frequency to remove (e.g., mains interference in UK)
NOTCH_Q = 30.0        # Quality factor

# Smoothing and detection parameters
MOVING_AVERAGE_WINDOW = 4         # Default window for the general moving_average function
DETECTION_SMOOTHING_WINDOW = 3      # Window size used for smoothing in the PID detection functions
DEPRESSION_THRESHOLD = 0.5          # Threshold for depressed values (50% reduction)
BAND_FRACTION_REQUIRED = 0.6        # At least 60% of bands must meet the condition

# Recovery criteria for per-band detection
RECOVERY_THRESHOLD = 0.9            # Average recovery threshold in the recovery window
SPIKE_THRESHOLD = 2.0               # Any value above this is considered a spike
SPIKE_WAIT_TIME = 5                 # Wait time (in seconds) after a spike to delay recovery detection

# Parameters for total power detection
MAD_MULTIPLIER = 1.5                # Multiplier used to compute the total power threshold from MAD
RECOVERY_WINDOW = 5                 # Window size (in seconds) for smoothing recovery values

# Baseline parameters
BASELINE_DURATION = 200             # Duration (in seconds) of the baseline period if baseline_end is not provided

#%%====================================================================
# Notch Filter Function
#====================================================================
def apply_notch_filter(data, fs=SAMPLE_RATE, freq=NOTCH_FREQ, Q=NOTCH_Q):
    b, a = iirnotch(w0=freq, Q=Q, fs=fs)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

#%%====================================================================
# Load Data from .h5 File
#====================================================================
def load_data_from_h5(file_path, group_path):
    with h5py.File(file_path, 'r') as h5file:
        data = h5file[group_path]['data'][:]
    return data

#%%====================================================================
# Compute Power Spectral Density (PSD) per Second
#====================================================================
def compute_psd_per_second(data, sample_rate=SAMPLE_RATE, bands=None):
    if bands is None:
        bands = DEFAULT_BANDS
    num_seconds = len(data) // sample_rate
    psd_values = {band: [] for band in bands}
    total_power = []
    for sec in range(num_seconds):
        start = sec * sample_rate
        end = start + sample_rate
        segment = data[start:end]
        freqs, psd = welch(segment, fs=sample_rate, nperseg=sample_rate)
        mean_band_powers = []
        for band, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.mean(psd[idx])
            psd_values[band].append(band_power)
            mean_band_powers.append(band_power)
        total_power.append(np.mean(mean_band_powers))
    return psd_values, total_power

#%%====================================================================
# Compute Baseline Statistics
#====================================================================
def compute_baseline_stats(data, sample_rate=SAMPLE_RATE, bands=None, baseline_start=0, baseline_end=None, baseline_duration=BASELINE_DURATION):
    if bands is None:
        bands = DEFAULT_BANDS
    if baseline_end is not None:
        start_idx = int(baseline_start * sample_rate)
        end_idx = int(baseline_end * sample_rate)
    else:
        start_idx = int(baseline_start * sample_rate)
        end_idx = int((baseline_start + baseline_duration) * sample_rate)
    baseline_data = data[start_idx:end_idx]
    baseline_psd_values, baseline_total_power = compute_psd_per_second(baseline_data, sample_rate, bands)
    
    # Print first 5 baseline PSD values
    print("\n=== Baseline PSD Values (First 5s) ===")
    for band, values in baseline_psd_values.items():
        print(f"{band}: {values[:5]}")
    
    mean_baseline_psd = {band: np.mean(baseline_psd_values[band]) for band in baseline_psd_values}
    normalized_baseline = {}
    mad_baseline_psd = {}
    for band in baseline_psd_values:
        normalized_values = [x / mean_baseline_psd[band] for x in baseline_psd_values[band]]
        normalized_baseline[band] = normalized_values
        mad_baseline_psd[band] = 1.5 * np.median(np.abs(np.array(normalized_values) - np.median(normalized_values)))
    num_seconds_baseline = len(next(iter(normalized_baseline.values())))
    baseline_normalized_total_power = [
        np.mean([normalized_baseline[band][t] for band in normalized_baseline])
        for t in range(num_seconds_baseline)
    ]
    mean_total_power_norm = np.mean(baseline_normalized_total_power)
    mad_total_power_norm = 1.5 * np.median(np.abs(np.array(baseline_normalized_total_power) - np.median(baseline_normalized_total_power)))
    
    print("\n=== Baseline Statistics (Normalized) ===")
    for band in mean_baseline_psd:
        print(f"{band}: Raw Mean = {mean_baseline_psd[band]:.4f}, Normalized MAD = {mad_baseline_psd[band]:.4f}")
    print(f"Overall Normalized Total Power: Mean = {mean_total_power_norm:.4f}, MAD = {mad_total_power_norm:.4f}")
    
    return mean_baseline_psd, mad_baseline_psd, mean_total_power_norm, mad_total_power_norm

#%%====================================================================
# Normalize PSD Values Relative to Baseline (Per-Band)
#====================================================================
def normalize_psd(psd_values, mean_baseline_psd):
    return {band: [x / mean_baseline_psd[band] for x in psd_values[band]] for band in psd_values}

#%%====================================================================
# Moving Average Filter (Reusable)
#====================================================================
def moving_average(data, window_size=MOVING_AVERAGE_WINDOW):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

#%%====================================================================
# Detect Post-Ictal Depression (Per-Band Method)
#====================================================================
def detect_post_ictal_depression(normalized_psd, seizure_duration, min_duration=8, sustained_duration=20, debug=True, extra_smoothing=False):
    """
    Detects post-ictal depression using per-band normalized PSD values with spike-insensitive averaging.
    Prints:
      - First 5 normalized PSD values per band and the depression threshold.
      - If a PID is detected: the time point and per-band adjusted means.
      - If no PID is detected: a message.
      - For recovery, only the final recovery time point with overall recovery mean,
        the per-band window means, and the count of bands recovered.
    """
    post_ictal_start = None
    post_ictal_end = None

    if not normalized_psd or not isinstance(normalized_psd, dict):
        raise ValueError("`normalized_psd` is empty or incorrectly formatted.")
    
    # Print initial input values.
    if debug:
        print("\n=== Normalized PSD Values (First 5 Points) per Band ===")
        for band, values in normalized_psd.items():
            print(f"{band:15s}: {values[:5]}")
        print(f"Depression Threshold (per band): {DEPRESSION_THRESHOLD:.4f}")
        print("-" * 40)

    # Apply smoothing if enabled.
    if extra_smoothing:
        smoothing_window = DETECTION_SMOOTHING_WINDOW
        processed_psd = {
            band: np.convolve(np.array(normalized_psd[band]), np.ones(smoothing_window)/smoothing_window, mode='valid')
            for band in normalized_psd
        }
    else:
        smoothing_window = 1
        processed_psd = {band: np.array(normalized_psd[band]) for band in normalized_psd}
    
    processed_length = len(next(iter(processed_psd.values())))
    start_idx = max(0, seizure_duration - (smoothing_window - 1))
    num_bands = len(normalized_psd)
    
    # PID Start Detection (spike-insensitive averaging)
    for t in range(start_idx, processed_length - min_duration + 1):
        depressed_band_count = 0
        for band in normalized_psd:
            window = processed_psd[band][t:t+min_duration]
            non_spike_values = window[window <= SPIKE_THRESHOLD]
            if len(non_spike_values) == 0:
                continue
            adjusted_mean = np.mean(non_spike_values)
            if adjusted_mean < DEPRESSION_THRESHOLD:
                depressed_band_count += 1
        if depressed_band_count >= int(BAND_FRACTION_REQUIRED * num_bands):
            post_ictal_start = t
            if debug:
                print(f"🛑 PID Start Detected at Time {t} sec")
                for band in normalized_psd:
                    window = processed_psd[band][t:t+min_duration]
                    non_spike_values = window[window <= SPIKE_THRESHOLD]
                    adjusted_mean = np.mean(non_spike_values) if non_spike_values.size > 0 else float('nan')
                    print(f"    {band:15s}: Adjusted Mean = {adjusted_mean:.4f}")
                print("-" * 40)
            break
    if post_ictal_start is None:
        if debug:
            print("No sustained depression detected for the minimum duration (per-band).")
        return None, None

    # PID Recovery Detection (only print the final recovery result)
    for t in range(post_ictal_start + min_duration, processed_length - sustained_duration + 1):
        if any(processed_psd[band][t] > SPIKE_THRESHOLD for band in normalized_psd):
            continue
        
        recovered_bands = {}
        for band in normalized_psd:
            window_mean = np.mean(processed_psd[band][t:t+sustained_duration])
            recovered_bands[band] = window_mean
        count_recovered = sum(1 for v in recovered_bands.values() if v > RECOVERY_THRESHOLD)
        if count_recovered >= int(BAND_FRACTION_REQUIRED * num_bands):
            post_ictal_end = t
            if debug:
                overall_recovery = np.mean(list(recovered_bands.values()))
                print(f"[Recovery] PID End Detected at Time {t} sec")
                print(f"Overall Recovery Mean = {overall_recovery:.4f}, Bands Recovered = {count_recovered}/{num_bands}")
                for band, val in recovered_bands.items():
                    print(f"    {band:15s}: Window Mean = {val:.4f}")
                print("=" * 40)
            break

    if post_ictal_end is None:
        if debug:
            print("Sustained recovery not found after PID start (per-band).")
        return None, None

    if (post_ictal_end - post_ictal_start) < min_duration:
        if debug:
            print("PID duration is shorter than the minimum required (per-band).")
        return None, None

    return post_ictal_start, post_ictal_end

#%%====================================================================
# Detect Post-Ictal Depression (Total Power Method)
#====================================================================
def detect_post_ictal_depression_total(total_power, mean_total_power, mad_total_power, seizure_duration,
                                        mad_multiplier=MAD_MULTIPLIER, min_duration=8, sustained_duration=20, debug=True):
    """
    Detects post-ictal depression based on overall total normalized power.
    Prints:
      - First 5 total power values and the computed threshold.
      - If a PID is detected: the time point and the total power value.
      - During recovery: only the final recovery time point with the overall recovery mean and total power value.
    """
    total_power_threshold = 1.0 - (mad_multiplier * mad_total_power)
    if debug:
        print("\n=== Total Power Values (First 5 Points) ===")
        print(total_power[:5])
        print(f"Overall Normalized Total Power Threshold: {total_power_threshold:.4f}")
        print("-" * 40)
    
    post_ictal_start = None
    post_ictal_end = None

    # PID Start Detection (Total Power Based)
    for t in range(seizure_duration, len(total_power) - min_duration + 1):
        candidate_window = total_power[t:t+min_duration]
        if np.all(np.array(candidate_window) < total_power_threshold):
            post_ictal_start = t
            if debug:
                print(f"🛑 PID Start Detected (Total Power) at Time {t} sec")
                print(f"    Total Power at this point: {total_power[t]:.4f}")
                print("-" * 40)
            break
    if post_ictal_start is None:
        if debug:
            print("No sustained depression detected for the minimum duration (total power-based).")
        return None, None

    # PID Recovery Detection (only print final recovery result)
    for t in range(post_ictal_start + min_duration, len(total_power)):
        recent_mean = np.mean(total_power[max(0, t-5):t])
        if total_power[t] > 1.8 * recent_mean:
            continue
        
        if t + sustained_duration <= len(total_power):
            window = total_power[t:t+sustained_duration]
        else:
            window = total_power[t:]
        recovery_values = np.convolve(window, np.ones(RECOVERY_WINDOW)/RECOVERY_WINDOW, mode='valid')
        if np.all(recovery_values > total_power_threshold):
            post_ictal_end = t
            if debug:
                overall_recovery = np.mean(window)
                print(f"[Recovery] (Total Power) PID End Detected at Time {t} sec")
                print(f"    Overall Recovery Mean = {overall_recovery:.4f}")
                print(f"    Total Power at this point: {total_power[t]:.4f}")
                print("-" * 40)
            break

    if post_ictal_end is None:
        if debug:
            print("Sustained recovery not found after PID start (total power-based).")
        return None, None

    if (post_ictal_end - post_ictal_start) < min_duration:
        if debug:
            print("PID duration is shorter than the minimum required (total power-based).")
        return None, None

    return post_ictal_start, post_ictal_end


#%%====================================================================
# Plotting Functions
#====================================================================
def plot_ecog_1hour(data, sample_rate, save_fig=False, filename=None):
    plt.figure(figsize=(12, 6), facecolor='none')
    time_axis = np.arange(len(data)) / sample_rate
    plt.plot(time_axis, data, color='black', linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("ECoG 1 Hour")
    # Remove top and right spines
    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_fig:
        if filename is None:
            filename = "ECoG_1_Hour.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def plot_ecog(data, sample_rate, seizure_start, seizure_end, save_fig=False, filename=None):
    plt.figure(figsize=(12, 6), facecolor='none')
    time_axis = np.arange(len(data)) / sample_rate
    plt.plot(time_axis, data, color='black', linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("ECoG for Selected Seizure/Post-Ictal Depression Period")
    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_fig:
        if filename is None:
            filename = "ECoG_Selected_Segment.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def plot_post_ictal_detection(time_axis, normalized_psd, post_ictal_start, post_ictal_end, save_fig=False, filename=None):
    plt.figure(figsize=(12, 6), facecolor='none')
    for band in normalized_psd:
        plt.plot(time_axis, normalized_psd[band], label=band, alpha=0.7)
    if post_ictal_start is not None:
        plt.axvline(post_ictal_start, color='red', linestyle='--', label='Post-Ictal Start')
    if post_ictal_end is not None:
        plt.axvline(post_ictal_end, color='green', linestyle='--', label='Post-Ictal End')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized PSD')
    plt.legend()
    plt.title('Post-Ictal Depression Detection')
    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_fig:
        if filename is None:
            filename = "Post_Ictal_Depression_Detection.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def plot_post_ictal_detection_total(time_axis, total_power, post_ictal_start, post_ictal_end, save_fig=False, filename=None):
    plt.figure(figsize=(12, 6), facecolor='none')
    plt.plot(time_axis, total_power, label='Total Power (Normalized)', color='blue', linewidth=1)
    if post_ictal_start is not None:
        plt.axvline(post_ictal_start, color='red', linestyle='--', label='Post-Ictal Start')
    if post_ictal_end is not None:
        plt.axvline(post_ictal_end, color='green', linestyle='--', label='Post-Ictal End')
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Total Power")
    plt.title("Post-Ictal Depression Detection (Total Power-Based)")
    plt.legend()
    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_fig:
        if filename is None:
            filename = "Post_Ictal_Depression_Total_Power.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def plot_ecog_with_psd(ecog_segment, sample_rate, selected_psd_values, save_fig=False, filename=None):
    """
    Plots a 200-second ECoG segment and overlays the per-band normalized PSD traces.
    
    Parameters:
      - ecog_segment: 1D NumPy array containing the 200-second ECoG data.
      - sample_rate: Sampling rate of the ECoG data.
      - selected_psd_values: Dictionary of normalized PSD values per band (1 value per second).
      - save_fig: Boolean flag to save the figure.
      - filename: Filename for saving the figure.
    
    The function uses twin axes:
      - Left y-axis shows the raw ECoG amplitude.
      - Right y-axis shows the per-band normalized PSD values.
    """
   
    # Create time axis for the ECoG segment (in seconds)
    time_axis_ecog = np.arange(len(ecog_segment)) / sample_rate
    
    # Create time axis for the PSD (each value represents one second)
    time_axis_psd = np.arange(len(next(iter(selected_psd_values.values()))))
    
    # Create the figure and primary axis for ECoG signal
    fig, ax1 = plt.subplots(figsize=(12, 6), facecolor='none')
    ax1.plot(time_axis_ecog, ecog_segment, color='black', linewidth=0.5, label='ECoG')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("ECoG Amplitude (µV)", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Make background transparent and remove top/right spines
    ax1.set_facecolor('none')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Create a secondary y-axis for normalized PSD values
    ax2 = ax1.twinx()
    for band, psd_values in selected_psd_values.items():
        ax2.plot(time_axis_psd, psd_values, label=band, alpha=0.7)
    ax2.set_ylabel("Normalized PSD", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_facecolor('none')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title("200-Second ECoG Segment with Overlaid Per-Band Normalized PSD")
    
    if save_fig:
        if filename is None:
            filename = "ECoG_with_PSD.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

#%%====================================================================
# Export to excel
#====================================================================
def export_to_excel(normalized_psd, total_power, post_ictal_band, post_ictal_band_end, 
                    post_ictal_total, post_ictal_total_end, filename, h5_filepath, h5_hierarchy):
    h5_filename = os.path.basename(h5_filepath)
    tid = h5_hierarchy.split('/')[-1] if '/' in h5_hierarchy else h5_hierarchy

    # Create full PSD DataFrame from normalized PSD values.
    df_full_psd = pd.DataFrame(normalized_psd)
    df_full_psd.insert(0, 'Time Index', range(len(df_full_psd)))
    df_full_psd.insert(0, '.h5 File', h5_filename)
    df_full_psd.insert(0, 'TID', tid)

    # Create PID Band-Based DataFrame and compute statistics after filtering out spikes.
    if post_ictal_band is not None and post_ictal_band_end is not None:
        df_pid_band = df_full_psd.iloc[post_ictal_band:post_ictal_band_end].copy()
        # Apply filtering only to numeric columns.
        numeric_cols = df_pid_band.select_dtypes(include=[np.number]).columns
        df_pid_band_filtered = df_pid_band.copy()
        df_pid_band_filtered[numeric_cols] = df_pid_band_filtered[numeric_cols].where(df_pid_band_filtered[numeric_cols] <= SPIKE_THRESHOLD)
        mean_psd_per_band = df_pid_band_filtered.mean(numeric_only=True)
        std_psd_per_band = df_pid_band_filtered.std(numeric_only=True)
        sem_psd_per_band = df_pid_band_filtered.sem(numeric_only=True)
    else:
        df_pid_band = pd.DataFrame()
        mean_psd_per_band = None
        std_psd_per_band = None
        sem_psd_per_band = None

    # Create PID Total Power-Based DataFrame and compute statistics after filtering out spikes.
    if post_ictal_total is not None and post_ictal_total_end is not None:
        df_pid_total = pd.DataFrame({'Total Power': total_power[post_ictal_total:post_ictal_total_end]})
        df_pid_total.insert(0, 'Time Index', range(post_ictal_total, post_ictal_total_end))
        df_pid_total.insert(0, '.h5 File', h5_filename)
        df_pid_total.insert(0, 'TID', tid)
        # Total power column is numeric; apply filtering directly.
        df_pid_total_filtered = df_pid_total.copy()
        df_pid_total_filtered['Total Power'] = df_pid_total_filtered['Total Power'].where(df_pid_total_filtered['Total Power'] <= SPIKE_THRESHOLD)
        mean_total_power_agg = df_pid_total_filtered['Total Power'].mean()
        std_total_power_agg = df_pid_total_filtered['Total Power'].std()
        sem_total_power_agg = df_pid_total_filtered['Total Power'].sem()
    else:
        df_pid_total = pd.DataFrame()
        mean_total_power_agg = None
        std_total_power_agg = None
        sem_total_power_agg = None

    # Build a summary dictionary that includes Mean, SD, and SEM.
    summary_data = {}
    summary_index = ["Mean", "SD", "SEM"]

    if mean_psd_per_band is not None and std_psd_per_band is not None and sem_psd_per_band is not None:
        for band in normalized_psd.keys():
            summary_data[band] = [mean_psd_per_band[band], std_psd_per_band[band], sem_psd_per_band[band]]

    if mean_total_power_agg is not None and std_total_power_agg is not None and sem_total_power_agg is not None:
        summary_data["Total Power"] = [mean_total_power_agg, std_total_power_agg, sem_total_power_agg]

    summary_df = pd.DataFrame(summary_data, index=summary_index) if summary_data else pd.DataFrame()

    with pd.ExcelWriter(filename) as writer:
        df_full_psd.to_excel(writer, sheet_name='Full PSD', index=False)
        if not df_pid_band.empty:
            df_pid_band.to_excel(writer, sheet_name='PID Band-Based', index=False)
        if not df_pid_total.empty:
            df_pid_total.to_excel(writer, sheet_name='PID Total Power-Based', index=False)
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='PID Summary')

    print(f"✅ Data exported to Excel successfully at {filename}")

#%%====================================================================
# Main Execution
#====================================================================
if __name__ == '__main__':
    # Define parameters that change per file in one place:
    h5_filepath =  '/Volumes/KH_PhD_01/P1/GRE_seizure_files/Seizure_h5/2208_TID58/M1661122630_2022-08-21-23-57-10_tids_[58].h5'
    h5_hierarchy = 'M1661122630/58'
    sample_rate = SAMPLE_RATE

    # Define a single seizure duration (in seconds) to be used everywhere
    SEIZURE_DURATION = 93  # Adjust as needed

    # Load data using the flexible function
    data = load_data_from_h5(h5_filepath, h5_hierarchy)
    
    # --- Apply Notch Filter to Remove 50 Hz Mains Interference ---
    APPLY_NOTCH_FILTER = True
    if APPLY_NOTCH_FILTER:
        data = apply_notch_filter(data, fs=sample_rate, freq=NOTCH_FREQ, Q=NOTCH_Q)
        print("Applied notch filter at 50 Hz to remove mains interference.")

    # Compute PSD for full 1-hour data (both individual band PSDs & total power)
    full_psd_values, full_total_power = compute_psd_per_second(data, sample_rate)

    # Compute baseline statistics using Approach B:
    baseline_start = 400      # Adjust as needed
    baseline_end = 600 # If None, BASELINE_DURATION seconds are used
    mean_baseline_psd, mad_baseline_psd, mean_total_power_norm, mad_total_power_norm = compute_baseline_stats(
        data, sample_rate, baseline_start=baseline_start, baseline_end=baseline_end
    )

    # Normalize PSD for full 1-hour data (per band using Approach B)
    normalized_psd = normalize_psd(full_psd_values, mean_baseline_psd)
    
    # Compute overall normalized total power by averaging per-band normalized values per second.
    num_seconds = len(next(iter(normalized_psd.values())))
    normalized_total_power = [
        np.mean([normalized_psd[band][t] for band in normalized_psd])
        for t in range(num_seconds)
    ]

    # Extract 200s seizure + post-ictal period
    seizure_start = 1434 * sample_rate
    seizure_end = 1634 * sample_rate
    selected_segment = data[seizure_start:seizure_end]
    
    # Extract normalized PSD and total power for the selected period
    selected_psd_values = {
        band: normalized_psd[band][seizure_start // sample_rate:seizure_end // sample_rate]
        for band in normalized_psd
    }
    selected_total_power = normalized_total_power[seizure_start // sample_rate:seizure_end // sample_rate]

    # Detect PID using Total Power-Based Method; pass SEIZURE_DURATION for consistency
    post_ictal_start_total, post_ictal_end_total = detect_post_ictal_depression_total(
        selected_total_power, mean_total_power_norm, mad_total_power_norm, seizure_duration=SEIZURE_DURATION
    )

    # Detect PID using Per-Band Method; pass SEIZURE_DURATION for consistency
    post_ictal_start_band, post_ictal_end_band = detect_post_ictal_depression(
        selected_psd_values, seizure_duration=SEIZURE_DURATION, min_duration=8, sustained_duration=20, debug=True
    )

    # Plot the full 1-hour ECoG and the seizure/PID segment.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_filename = os.path.basename(h5_filepath).replace(".h5", "")
    
    ecog_1hr_filename = f"{h5_filename}_1hr_ECoG_{timestamp}.png"
    plot_ecog_1hour(data, sample_rate, save_fig=True, filename=ecog_1hr_filename)

    ecog_segment_filename = f"{h5_filename}_Seizure_Selected_{timestamp}.png"
    plot_ecog(selected_segment, sample_rate, seizure_start, seizure_end, save_fig=True, filename=ecog_segment_filename)

    post_ictal_detection_filename = f"{h5_filename}_Post_Ictal_Detection_{timestamp}.png"
    time_axis_psd = np.arange(len(selected_psd_values[list(selected_psd_values.keys())[0]]))
    plot_post_ictal_detection(time_axis_psd, selected_psd_values, post_ictal_start_band, post_ictal_end_band,
                              save_fig=True, filename=post_ictal_detection_filename)
    
    post_ictal_total_filename = f"{h5_filename}_Post_Ictal_Total_Power_{timestamp}.png"
    time_axis_total = np.arange(len(selected_total_power))
    plot_post_ictal_detection_total(time_axis_total, selected_total_power, post_ictal_start_total, post_ictal_end_total,
                                    save_fig=True, filename=post_ictal_total_filename)

    # Plot an additional trace: Overlay per-band PSD traces onto the 200-second ECoG segment.
    overlay_filename = f"{h5_filename}_ECoG_with_PSD_{timestamp}.png"
    plot_ecog_with_psd(selected_segment, sample_rate, selected_psd_values, save_fig=True, filename=overlay_filename)

    tid = h5_hierarchy.split("/")[-1]
    output_excel_filename = f"{h5_filename}_TID{tid}_post_ictal_analysis_{timestamp}.xlsx"
    print(f"✅ Saving Excel as: {output_excel_filename}")

    export_to_excel(
        selected_psd_values, selected_total_power, 
        post_ictal_start_band, post_ictal_end_band, 
        post_ictal_start_total, post_ictal_end_total, 
        output_excel_filename, h5_filepath, h5_hierarchy
    )

