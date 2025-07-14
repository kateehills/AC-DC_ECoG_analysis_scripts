#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:29:28 2025

@author: katehills

Modified for Seizure Detection by using amplitude and energy (total power) thresholds.
Threshold criteria: 
    - Amplitude: moving average of absolute signal > 2× baseline amplitude.
    - Total power: normalized total power > 2× baseline normalized total power.
A sustained period of at least 8 seconds is required for detection.
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
MOVING_AVERAGE_WINDOW = 4         # For moving average filtering
MIN_DURATION = 8                  # Minimum sustained duration (in seconds) for detection
SUSTAINED_DURATION = 20           # For potential recovery detection (if later developed)

# Baseline parameters
BASELINE_DURATION = 200           # 200 seconds of baseline data
# In our example, we will use baseline_start=400 and baseline_end=600

#====================================================================
# Notch Filter Function
#====================================================================
def apply_notch_filter(data, fs=SAMPLE_RATE, freq=NOTCH_FREQ, Q=NOTCH_Q):
    b, a = iirnotch(w0=freq, Q=Q, fs=fs)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

#====================================================================
# Load Data from .h5 File
#====================================================================
def load_data_from_h5(file_path, group_path):
    with h5py.File(file_path, 'r') as h5file:
        data = h5file[group_path]['data'][:]
    return data

#====================================================================
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

#====================================================================
# Compute Baseline Statistics (including amplitude)
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
    
    # Compute PSD values for the baseline period
    baseline_psd_values, baseline_total_power = compute_psd_per_second(baseline_data, sample_rate, bands)
    
    # Compute baseline amplitude (mean absolute value)
    baseline_amplitude = np.mean(np.abs(baseline_data))
    
    # Compute mean PSD per band and normalized MAD values
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
    
    print("\n=== Baseline PSD Statistics (Normalized) ===")
    for band in mean_baseline_psd:
        print(f"{band}: Raw Mean = {mean_baseline_psd[band]:.4f}, Normalized MAD = {mad_baseline_psd[band]:.4f}")
    print(f"Overall Normalized Total Power: Mean = {mean_total_power_norm:.4f}, MAD = {mad_total_power_norm:.4f}")
    
    return mean_baseline_psd, mad_baseline_psd, mean_total_power_norm, mad_total_power_norm, baseline_amplitude

#====================================================================
# Normalize PSD Values Relative to Baseline (Per-Band)
#====================================================================
def normalize_psd(psd_values, mean_baseline_psd):
    return {band: [x / mean_baseline_psd[band] for x in psd_values[band]] for band in psd_values}

#====================================================================
# Moving Average Filter (Reusable)
#====================================================================
def moving_average(data, window_size=MOVING_AVERAGE_WINDOW):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

#====================================================================
# Detect Seizure Onset Based on Total Power
#====================================================================
def detect_seizure_onset_total(total_power, baseline_total_power, seizure_duration,
                               min_duration=MIN_DURATION, debug=True):
    """
    Detects seizure onset based on an increase in overall normalized total power.
    A seizure is detected if the total power exceeds 2x the baseline total power for at least min_duration seconds.
    """
    total_power_threshold = 2.0 * baseline_total_power
    if debug:
        print("\n=== Total Power-Based Seizure Detection ===")
        #print("First 5 total power values:", total_power[:5])
        print(f"Total Power Threshold for Seizure Onset: {total_power_threshold:.4f}")
        print("-" * 40)
    
    seizure_onset = None
    for t in range(seizure_duration, len(total_power) - min_duration + 1):
        candidate_window = total_power[t:t+min_duration]
        if np.all(np.array(candidate_window) > total_power_threshold):
            seizure_onset = t
            if debug:
                #print(f"🛑 Seizure Onset Detected (Total Power) at Time {t} sec")
                print(f"    Total Power at this point: {total_power[t]:.4f}")
                print("-" * 40)
            break

    if seizure_onset is None and debug:
        print("No sustained seizure onset detected (total power-based).")
    
    return seizure_onset

#====================================================================
# Detect Seizure Onset Based on Amplitude
#====================================================================
def detect_seizure_onset_amplitude(data, sample_rate, baseline_amplitude, min_duration=MIN_DURATION, debug=True, required_fraction=0.5):
    """
    Detects seizure onset based on amplitude.
    A seizure is detected if at least 'required_fraction' of the samples in a window of 
    'min_duration' seconds (converted to samples) exceed 2x the baseline amplitude.
    """
    amplitude_envelope = np.abs(data)
    
    # if debug:
    #     output_filename = "amplitude_values.txt"
    #     with open(output_filename, "w") as f:
    #         for value in amplitude_envelope:
    #             f.write(f"{value}\n")
    #     print(f"\n=== All Amplitude Envelope Values Written to {output_filename} ===")
    
    smoothed_envelope = moving_average(amplitude_envelope, window_size=MOVING_AVERAGE_WINDOW)
    amplitude_threshold = 2.0 * baseline_amplitude
    if debug:
        print("\n=== Amplitude-Based Seizure Detection ===")
        print(f"Amplitude Threshold: {amplitude_threshold:.4f}")
    
    seizure_onset = None
    min_duration_samples = min_duration * sample_rate
    for t in range(0, len(smoothed_envelope) - min_duration_samples + 1):
        window = smoothed_envelope[t:t+min_duration_samples]
        if np.mean(window > amplitude_threshold) >= required_fraction:
            seizure_onset = t
            #if debug:
                #print(f"🛑 Seizure Onset Detected (Amplitude) at sample {t} (~{t/sample_rate:.2f} sec)")
            break
            
    if seizure_onset is None and debug:
        print("No sustained seizure onset detected (amplitude-based).")
    
    return seizure_onset

#====================================================================
# Detect Seizure Termination Based on Total Power
#====================================================================
def detect_seizure_termination_total(total_power, baseline_total_power_norm, seizure_onset, min_duration=MIN_DURATION, debug=True):
    termination = None
    for t in range(seizure_onset, len(total_power) - min_duration + 1):
        candidate_window = total_power[t:t + min_duration]
        # fraction_below = np.mean(candidate_window < baseline_total_power_norm)
        # if fraction_below > 0.7:
        #     termination = t
        if np.all(np.array(candidate_window) < baseline_total_power_norm):
            termination = t
            if debug:
                print(f"🛑 Seizure Termination Detected (Total Power) at {t} sec")
            break
    if termination is None and debug:
        print("No sustained seizure termination detected (total power-based).")
    return termination

#====================================================================
# Detect Seizure Termination Based on Amplitude
#====================================================================
def detect_seizure_termination_amplitude(data, sample_rate, baseline_amplitude, seizure_onset, min_duration=MIN_DURATION, debug=False, required_fraction=0.9):
    # Immediately return if no seizure onset was detected.
    if seizure_onset is None:
        if debug:
            print("No seizure onset detected; skipping amplitude termination detection.")
        return None
        
    amplitude_envelope = np.abs(data)
    smoothed_envelope = moving_average(amplitude_envelope, window_size=MOVING_AVERAGE_WINDOW)
    amplitude_threshold = 2.0 * baseline_amplitude
    if debug:
        print("\n=== Amplitude-Based Seizure Termination Detection ===")
        print(f"Amplitude Threshold: {amplitude_threshold:.4f}")
    
    termination = None
    min_duration_samples = int(min_duration * sample_rate)
    for t in range(seizure_onset, len(smoothed_envelope) - min_duration_samples + 1):
        window = smoothed_envelope[t:t+min_duration_samples]
        fraction_below = np.mean(window < amplitude_threshold)
        if fraction_below >= required_fraction:
            termination = t
            if debug:
                print(f"🛑 Seizure Termination Detected (Amplitude) at sample {t} (~{t/sample_rate:.2f} sec)")
            break
            
    if termination is None and debug:
        print("No sustained seizure termination detected (amplitude-based).")
    
    return termination
5
#====================================================================
# Plotting Functions
#====================================================================
def plot_ecog_1hour(data, sample_rate, save_fig=False, filename=None):
    plt.figure(figsize=(12, 6), facecolor='none')
    time_axis = np.arange(len(data)) / sample_rate
    plt.plot(time_axis, data, color='black', linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("ECoG 1 Hour")
    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_fig:
        if filename is None:
            filename = "ECoG_1_Hour.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def plot_ecog(data, sample_rate, start_idx, end_idx, save_fig=False, filename=None):
    plt.figure(figsize=(12, 6), facecolor='none')
    time_axis = np.arange(len(data)) / sample_rate
    plt.plot(time_axis, data, color='black', linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("ECoG Segment (Seizure/Selected Window)")
    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_fig:
        if filename is None:
            filename = "ECoG_Selected_Segment.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def plot_seizure_detection_total(time_axis, total_power, seizure_onset, seizure_termination=None, save_fig=False, filename=None):
    plt.figure(figsize=(12, 6), facecolor='none')
    plt.plot(time_axis, total_power, label='Normalized Total Power', color='blue', linewidth=1)
    if seizure_onset is not None:
        plt.axvline(seizure_onset, color='red', linestyle='--', label='Seizure Onset')
    if seizure_termination is not None:
        plt.axvline(seizure_termination, color='green', linestyle='--', label='Seizure Termination')
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Total Power")
    plt.title("Seizure Detection (Total Power-Based)")
    plt.legend()
    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_fig:
        if filename is None:
            filename = "Seizure_Detection_Total_Power.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    
def plot_seizure_detection_amplitude(smoothed_envelope, amplitude_threshold, seizure_onset, sample_rate, seizure_termination=None, save_fig=False, filename=None):
    time_axis = np.arange(len(smoothed_envelope)) / sample_rate
    plt.figure(figsize=(12, 6), facecolor='none')
    plt.plot(time_axis, smoothed_envelope, label='Smoothed Amplitude', color='green', linewidth=1)
    plt.axhline(y=amplitude_threshold, color='red', linestyle='--', label='Amplitude Threshold (2x Baseline)')
    if seizure_onset is not None:
        onset_time = seizure_onset / sample_rate
        plt.axvline(x=onset_time, color='blue', linestyle='--', label='Seizure Onset')
    if seizure_termination is not None:
        termination_time = seizure_termination / sample_rate
        plt.axvline(x=termination_time, color='green', linestyle='--', label='Seizure Termination')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Seizure Detection (Amplitude-Based)")
    plt.legend()
    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_fig:
        if filename is None:
            filename = "Seizure_Detection_Amplitude.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def plot_ecog_with_psd(ecog_segment, sample_rate, selected_psd_values, save_fig=False, filename=None):
    time_axis_ecog = np.arange(len(ecog_segment)) / sample_rate
    time_axis_psd = np.arange(len(next(iter(selected_psd_values.values()))))
    fig, ax1 = plt.subplots(figsize=(12, 6), facecolor='none')
    ax1.plot(time_axis_ecog, ecog_segment, color='black', linewidth=0.5, label='ECoG')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("ECoG Amplitude (µV)", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_facecolor('none')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = ax1.twinx()
    for band, psd_values in selected_psd_values.items():
        ax2.plot(time_axis_psd, psd_values, label=band, alpha=0.7)
    ax2.set_ylabel("Normalized PSD", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_facecolor('none')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.title("200-Second ECoG Segment with Overlaid Per-Band Normalized PSD")
    if save_fig:
        if filename is None:
            filename = "ECoG_with_PSD.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

#====================================================================
# Export to Excel
#====================================================================
def export_to_excel(normalized_psd, selected_total_power, smoothed_envelope, sample_rate,
                    seizure_onset_amplitude, seizure_onset_total, 
                    seizure_termination_amplitude, seizure_termination_total,
                    baseline_amplitude,
                    filename, h5_filepath, h5_hierarchy):
    # Extract basic metadata from the file path/hierarchy
    h5_filename = os.path.basename(h5_filepath)
    tid = h5_hierarchy.split('/')[-1] if '/' in h5_hierarchy else h5_hierarchy

    # ---------------------------
    # Full PSD DataFrame (for the entire recording)
    df_full_psd = pd.DataFrame(normalized_psd)
    df_full_psd.insert(0, 'Time (sec)', np.arange(len(df_full_psd)))
    df_full_psd.insert(0, '.h5 File', h5_filename)
    df_full_psd.insert(0, 'TID', tid)

    # ---------------------------
    # Seizure Total Power DataFrame (conditionally created)
    if seizure_onset_total is not None:
        if seizure_termination_total is not None:
            tp_term = seizure_termination_total
        else:
            tp_term = len(selected_total_power) - 1  # Use end of window if termination not detected
        seizure_total_power = selected_total_power[seizure_onset_total:tp_term+1]
        df_total_power = pd.DataFrame({
            "Time (sec)": np.arange(seizure_onset_total, tp_term+1),
            "Normalized Total Power": seizure_total_power
        })
    else:
        df_total_power = pd.DataFrame()  # Empty DataFrame if no onset is detected

    # ---------------------------
    # Seizure Smoothed Amplitude DataFrame (conditionally created)
    if seizure_onset_amplitude is not None:
        if seizure_termination_amplitude is not None:
            term_index = seizure_termination_amplitude
        else:
            term_index = len(smoothed_envelope) - 1
        seizure_smoothed_amp = smoothed_envelope[seizure_onset_amplitude:term_index+1]
        times_amp = np.arange(seizure_onset_amplitude, term_index+1) / sample_rate
        df_smoothed_amp = pd.DataFrame({
            "Time (sec)": times_amp,
            "Smoothed Amplitude": seizure_smoothed_amp
        })
    else:
        df_smoothed_amp = pd.DataFrame()

    # ---------------------------
    # Seizure Window Total Power DataFrame (full 200‑second window)
    window_total_power = selected_total_power  # full 200‑second window (one value per second)
    window_times = np.arange(len(window_total_power))
    df_window_total_power = pd.DataFrame({
         "Time (sec)": window_times,
         "Normalized Total Power": window_total_power
    })

    # ---------------------------
    # Selected PSD DataFrame (per-band PSD for selected 200‑second window)
    # Assume that the full normalized PSD arrays are per-second values.
    try:
        selected_start_sec = seizure_start_idx // sample_rate
        selected_end_sec = seizure_end_idx // sample_rate
    except NameError:
        selected_start_sec = 0
        selected_end_sec = len(next(iter(normalized_psd.values())))
    df_selected_psd = pd.DataFrame({band: normalized_psd[band][selected_start_sec:selected_end_sec]
                                    for band in normalized_psd})
    df_selected_psd.insert(0, "Time (sec)", np.arange(selected_start_sec, selected_end_sec))

    # ---------------------------
    # Seizure Summary DataFrame
    # Total Power Summary:
    if seizure_onset_total is not None:
        start_tp = seizure_onset_total
        if seizure_termination_total is not None:
            end_tp = seizure_termination_total
            duration_tp = end_tp - start_tp
        else:
            end_tp = np.nan
            duration_tp = np.nan
        mean_tp = np.mean(selected_total_power[seizure_onset_total:tp_term+1])
        std_tp = np.std(selected_total_power[seizure_onset_total:tp_term+1])
        sem_tp = std_tp / np.sqrt(len(selected_total_power[seizure_onset_total:tp_term+1]))
    else:
        start_tp = np.nan
        end_tp = np.nan
        duration_tp = np.nan
        mean_tp = np.nan
        std_tp = np.nan
        sem_tp = np.nan

    # Amplitude Summary:
    if seizure_onset_amplitude is not None:
        start_amp = seizure_onset_amplitude / sample_rate
        if seizure_termination_amplitude is not None:
            end_amp = seizure_termination_amplitude / sample_rate
            duration_amp = end_amp - start_amp
        else:
            end_amp = np.nan
            duration_amp = np.nan
        mean_amp = np.mean(seizure_smoothed_amp) if not df_smoothed_amp.empty else np.nan
        std_amp = np.std(seizure_smoothed_amp) if not df_smoothed_amp.empty else np.nan
        sem_amp = std_amp / np.sqrt(len(seizure_smoothed_amp)) if not df_smoothed_amp.empty else np.nan
    else:
        start_amp = np.nan
        end_amp = np.nan
        duration_amp = np.nan
        mean_amp = np.nan
        std_amp = np.nan
        sem_amp = np.nan

    # Calculate thresholds:
    total_power_threshold = 2.0   # For normalized total power (baseline normalized to ~1)
    amplitude_threshold = 2.0 * baseline_amplitude

    summary_data = {
        "Metric": ["Total Power", "Smoothed Amplitude"],
        "Threshold": [total_power_threshold, amplitude_threshold],
        "Seizure Start (sec)": [start_tp, start_amp],
        "Seizure End (sec)": [end_tp, end_amp],
        "Duration (sec)": [duration_tp, duration_amp],
        "Mean": [mean_tp, mean_amp],
        "Std Dev": [std_tp, std_amp],
        "SEM": [sem_tp, sem_amp]
    }
    df_summary = pd.DataFrame(summary_data)

    # ---------------------------
    # Write all DataFrames to separate sheets in the Excel file.
    with pd.ExcelWriter(filename) as writer:
        df_full_psd.to_excel(writer, sheet_name='Full PSD', index=False)
        df_summary.to_excel(writer, sheet_name='Seizure Summary', index=False)
        if not df_total_power.empty:
            df_total_power.to_excel(writer, sheet_name='Seizure Total Power', index=False)
        if not df_smoothed_amp.empty:
            df_smoothed_amp.to_excel(writer, sheet_name='Seizure Smoothed Amplitude', index=False)
        df_window_total_power.to_excel(writer, sheet_name='Window Total Power', index=False)
        df_selected_psd.to_excel(writer, sheet_name='Selected PSD', index=False)

#====================================================================
# Main Execution
#====================================================================
if __name__ == '__main__':
    # Define file paths and parameters (adjust as needed)
    h5_filepath = '/Volumes/KH_PhD_01/P1/GRE_seizure_files/Seizure_h5/2208_TID59/M1661036206_2022-08-20-23-56-46_tids_[59].h5'
    h5_hierarchy = 'M1661036206/59'
    sample_rate = SAMPLE_RATE
    SEIZURE_DURATION = 75  # In seconds; adjust as needed

    # Load data from .h5 file
    data = load_data_from_h5(h5_filepath, h5_hierarchy)
    
    # Optionally apply notch filter to remove mains interference
    APPLY_NOTCH_FILTER = True
    if APPLY_NOTCH_FILTER:
        data = apply_notch_filter(data, fs=sample_rate, freq=NOTCH_FREQ, Q=NOTCH_Q)
        print("Applied notch filter at 50 Hz to remove mains interference.")
    
    # Compute PSD for full data (1-hour assumed)
    full_psd_values, full_total_power = compute_psd_per_second(data, sample_rate)
    
    # Compute baseline statistics (using 200 seconds of baseline data; e.g., from 400 s to 600 s)
    baseline_start = 1100
    baseline_end = 1300
    mean_baseline_psd, mad_baseline_psd, mean_total_power_norm, mad_total_power_norm, baseline_amplitude = compute_baseline_stats(
        data, sample_rate, baseline_start=baseline_start, baseline_end=baseline_end
    )
    
    # Normalize PSD for full data (if needed for power-based detection)
    normalized_psd = normalize_psd(full_psd_values, mean_baseline_psd)
    num_seconds = len(next(iter(normalized_psd.values())))
    normalized_total_power = [
        np.mean([normalized_psd[band][t] for band in normalized_psd])
        for t in range(num_seconds)
    ]
    
    # Extract a selected 200-second window (for example, from 2300 sec to 2500 sec)
    seizure_start_idx = 2750 * sample_rate
    seizure_end_idx = 2950 * sample_rate
    selected_segment = data[seizure_start_idx:seizure_end_idx]
    selected_total_power = normalized_total_power[seizure_start_idx // sample_rate:seizure_end_idx // sample_rate]
    
    # Detect seizure onset using Total Power criteria
    seizure_onset_total = detect_seizure_onset_total(
        selected_total_power, mean_total_power_norm, seizure_duration=SEIZURE_DURATION
    )
    
    if seizure_onset_total is not None:
        seizure_termination_total = detect_seizure_termination_total(
            selected_total_power, mean_total_power_norm, seizure_onset_total, min_duration=MIN_DURATION, debug=False
        )
    else:
        seizure_termination_total = None
        print("No seizure onset (total power) detected; skipping termination detection for total power.")
    
    # Detect seizure onset using Amplitude criteria
    seizure_onset_amplitude = detect_seizure_onset_amplitude(
        selected_segment, sample_rate, baseline_amplitude
    )
    
    # Detect seizure termination using Amplitude criteria
    seizure_termination_amplitude = detect_seizure_termination_amplitude(
        selected_segment, sample_rate, baseline_amplitude, seizure_onset_amplitude, min_duration=MIN_DURATION, debug=True, required_fraction=0.9
    )

    # Print summary of onset and termination
    if seizure_onset_total is not None:
        print(f"✅ Seizure Onset (Total Power): {seizure_onset_total} sec")
    else:
        print("✅ Seizure Onset (Total Power): No onset detected.")
    
    if seizure_termination_total is not None:
        print(f"🛑 Seizure Termination (Total Power): {seizure_termination_total} sec")
    else:
        print("🛑 Seizure Termination (Total Power): No termination detected.")
    
    if seizure_onset_amplitude is not None:
        print(f"✅ Seizure Onset (Amplitude): sample {seizure_onset_amplitude} (~{seizure_onset_amplitude/sample_rate:.2f} sec)")
    else:
        print("✅ Seizure Onset (Amplitude): No sustained onset detected.")
    
    if seizure_termination_amplitude is not None:
        print(f"🛑 Seizure Termination (Amplitude): sample {seizure_termination_amplitude} (~{seizure_termination_amplitude/sample_rate:.2f} sec)")
    else:
        print("🛑 Seizure Termination (Amplitude): No sustained drop detected.")


    # Compute raw/smoothed amplitude envelope and threshold for plotting
    raw_amplitude = np.abs(selected_segment)
    amplitude_envelope = np.abs(selected_segment)
    smoothed_envelope = moving_average(amplitude_envelope, window_size=MOVING_AVERAGE_WINDOW)
    amplitude_threshold = 2.0 * baseline_amplitude
        
    # Generate filenames with timestamp for saving outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_filename = os.path.basename(h5_filepath).replace(".h5", "")
    
    ecog_1hr_filename = f"{h5_filename}_1hr_ECoG_{timestamp}.png"
    plot_ecog_1hour(data, sample_rate, save_fig=True, filename=ecog_1hr_filename)
    
    ecog_segment_filename = f"{h5_filename}_Seizure_Selected_{timestamp}.png"
    plot_ecog(selected_segment, sample_rate, seizure_start_idx, seizure_end_idx, save_fig=True, filename=ecog_segment_filename)
    
    seizure_total_filename = f"{h5_filename}_Seizure_Detection_Total_{timestamp}.png"
    time_axis_total = np.arange(len(selected_total_power))
    plot_seizure_detection_total(time_axis_total, selected_total_power, seizure_onset_total, seizure_termination_total,
                                 save_fig=True, filename=seizure_total_filename)
    
    overlay_filename = f"{h5_filename}_ECoG_with_PSD_{timestamp}.png"
    plot_ecog_with_psd(selected_segment, sample_rate, {band: normalized_psd[band][seizure_start_idx//sample_rate:seizure_end_idx//sample_rate] for band in normalized_psd},
                       save_fig=True, filename=overlay_filename)
    
    amp_filename = f"{h5_filename}_Seizure_Detection_Amplitude_{timestamp}.png"
    plot_seizure_detection_amplitude(smoothed_envelope, amplitude_threshold, seizure_onset_amplitude, sample_rate, 
                                     seizure_termination=seizure_termination_amplitude, save_fig=True, filename=amp_filename)

    # Export results to Excel with additional sheets for seizure period details.
    tid = h5_hierarchy.split("/")[-1]
    output_excel_filename = f"{h5_filename}_TID{tid}_seizure_analysis_{timestamp}_seizure.xlsx"
    print(f"✅ Saving Excel as: {output_excel_filename}")
    
    export_to_excel(
        normalized_psd, 
        selected_total_power, 
        smoothed_envelope,   # smoothed envelope of the amplitude
        sample_rate,
        seizure_onset_amplitude, 
        seizure_onset_total,
        seizure_termination_amplitude, 
        seizure_termination_total,
        baseline_amplitude,   # NEW parameter for threshold calculation
        output_excel_filename, 
        h5_filepath, 
        h5_hierarchy
    )
