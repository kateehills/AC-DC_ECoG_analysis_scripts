#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 16:21:12 2025

@author: katehills
"""

import h5py
import os
import matplotlib.pyplot as plt
import scipy.signal as signal

plt.close('all')

# Filtering options: set FILTER_TYPE to 'bandpass', 'lowpass', or 'none'
FILTER_TYPE = 'lowpass'
BANDPASS_LOW = 0.3    # Lower cutoff for bandpass filter (Hz)
BANDPASS_HIGH = 1 # Upper cutoff for bandpass filter or cutoff for lowpass (Hz)

# Input custom colors for each channel as hex codes (GraphPad Prism's Muted Rainbow for example)
channel_colors = ['firebrick', 'royalblue', 'forestgreen', 'mediumpurple'] #'royalblue'

def load_data_from_h5(file_path, group_path, dataset_name=None):
    """Load data from a .h5 file."""
    with h5py.File(file_path, 'r') as h5file:
        if dataset_name is None:
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        data = h5file[group_path]['data'][:]
    return data

def apply_filter(data, sample_rate, filter_type='bandpass', lowcut=0.5, highcut=45, order=4):
    """Apply a digital Butterworth filter to the signal.
    
    Parameters:
        data (ndarray): The raw signal data.
        sample_rate (int): Sampling rate in Hz.
        filter_type (str): Type of filter: 'bandpass', 'lowpass', or 'none'.
        lowcut (float): Lower cutoff frequency for bandpass filter.
        highcut (float): Upper cutoff frequency for bandpass filter (or cutoff for lowpass).
        order (int): Order of the filter.
    
    Returns:
        Filtered signal.
    """
    nyq = 0.5 * sample_rate
    if filter_type == 'bandpass':
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
    elif filter_type == 'lowpass':
        cutoff = highcut / nyq
        b, a = signal.butter(order, cutoff, btype='low')
    else:
        # If no filtering is required, return original data.
        return data
    return signal.filtfilt(b, a, data)

def plot_time_series(ax, data, sample_rate=512, color='k', linewidth=1, label=None, alpha=1.0):
    """
    Plot the time series electrophysiology data on a given subplot.
    
    Parameters:
        ax (Axes): The subplot where the data will be plotted.
        data (ndarray): The time series data to be plotted.
        sample_rate (int, optional): The sampling rate of the data. Default is 512 Hz.
        color (str, optional): Plot color. Default is 'k' (black).
        linewidth (float, optional): Line width. Default is 0.5.
        label (str, optional): Legend label.
        alpha (float, optional): Transparency level of the line (0: transparent, 1: opaque). Default is 1.0.
    """
    time = [i / sample_rate for i in range(len(data))]
    ax.plot(time, data, color=color, linewidth=linewidth, label=label, alpha=alpha)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (uV)')
    ax.grid(False)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Replace with the actual path to your .h5 file
h5_filepath = '/Volumes/KH_PhD_01/P2/GLT_GRE_pyecog2/TID41-4/E1718868279_2024-06-20-08-24-39_tids_[41, 42, 43, 44].h5'
parent_group = 'E1718868279'  # Parent group containing TIDs (e.g., '213', '214', '216')

# Open the .h5 file to get a list of TID groups
with h5py.File(h5_filepath, 'r') as h5file:
    tids = list(h5file[parent_group].keys())  # e.g., ['213', '214', '216']

tid_data_list = []  # to store (tid, filtered_data, color) for overlay plot

# Define the transparency level for each line
line_alpha = 0.8

# Loop over each TID, apply filtering, and generate an individual transparent plot
for i, tid in enumerate(tids):
    group_path = f"{parent_group}/{tid}"
    data = load_data_from_h5(h5_filepath, group_path)
    
    # Apply filtering to the signal
    filtered_data = apply_filter(data, sample_rate=512, filter_type=FILTER_TYPE, 
                                 lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH, order=4)
    
    # Assign a color for this TID using the custom color list.
    color = channel_colors[i % len(channel_colors)]
    
    # Store the filtered data for the overlay plot
    tid_data_list.append((tid, filtered_data, color))
    
    # Create a transparent figure and axis for the individual plot
    fig, ax = plt.subplots(1, 1, figsize=(17, 5), sharex=True, sharey=True)
    fig.patch.set_alpha(0)  # Transparent figure background
    ax.set_facecolor('none')  # Transparent axes background
    
    # Plot the filtered time series data with slight transparency for the line
    plot_time_series(ax, filtered_data, sample_rate=512, color=color, linewidth=1, label=f"TID {tid}", alpha=line_alpha)
    
    # Set title and axis limits
    fig.suptitle(f'TID {tid}_GRE (Filtered: {FILTER_TYPE})')
    plt.xlim(0,3600)
    ax.set_ylim(-3000, 2000)
    
    plt.tight_layout()
    
    # Save the individual plot as both PNG and SVG files, including the parent group and filter high in the filename
    png_filename = f"{parent_group}_TID_{tid}_GRE_{FILTER_TYPE}_{BANDPASS_HIGH}.png"
    svg_filename = f"{parent_group}_TID_{tid}_GRE_{FILTER_TYPE}_{BANDPASS_HIGH}.svg"
    plt.savefig(png_filename, transparent=True, bbox_inches='tight')
    plt.savefig(svg_filename, transparent=True, bbox_inches='tight')
    
    plt.show()
    print(f"Saved plots for TID {tid} as {png_filename} and {svg_filename}")

# Create an overlay plot that includes all TIDs with their respective colors
fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(17, 5), sharex=True, sharey=True)
fig_overlay.patch.set_alpha(0)
ax_overlay.set_facecolor('none')

for tid, filtered_data, color in tid_data_list:
    plot_time_series(ax_overlay, filtered_data, sample_rate=512, color=color, linewidth=1, label=f"TID {tid}", alpha=line_alpha)

ax_overlay.set_xlim(0,3600)
ax_overlay.set_ylim(-3000, 2000)
fig_overlay.suptitle(f'Overlay of All TIDs (Filtered: {FILTER_TYPE})')
plt.tight_layout()

# Save the overlay plot as both PNG and SVG files, including the parent group and filter high in the filename
overlay_png = f"{parent_group}_Overlay_All_TIDs_{FILTER_TYPE}_{BANDPASS_HIGH}.png"
overlay_svg = f"{parent_group}_Overlay_All_TIDs_{FILTER_TYPE}_{BANDPASS_HIGH}.svg"
plt.savefig(overlay_png, transparent=True, bbox_inches='tight')
plt.savefig(overlay_svg, transparent=True, bbox_inches='tight')

plt.show()
print(f"Saved overlay plot as {overlay_png} and {overlay_svg}")
