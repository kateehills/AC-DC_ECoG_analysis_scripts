#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:54:47 2024

@author: katehills
"""

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import matplotlib.gridspec as gridspec

plt.close('all')

def load_data_from_h5(file_path, group_path, dataset_name=None):
    """Load data from a .h5 file."""
    with h5py.File(file_path, 'r') as h5file:
        if dataset_name is None:
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        data = h5file[group_path]['data'][:]
    return data

def plot_time_series(ax, data, sample_rate=256, color='k', linewidth=0.5, label=None):
    """
    Plot the time series electrophysiology data on a given subplot.
    """
    time = np.arange(len(data)) / sample_rate
    ax.plot(time, data, color=color, linewidth=linewidth, label=label)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (uV)')
    ax.grid(False)
    if label is not None:
        ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_time_frequency(ax, data, sample_rate=256, xlim=None, freq_max=120, n_cycles=7,
                        cmap='viridis', vmin=None, vmax=None):
    """
    Compute and plot a time-frequency representation using MNE-Python's Morlet wavelets.
    Returns the QuadMesh object so that a colorbar can be added externally.
    """
    # If xlim is provided, slice the data accordingly
    if xlim is not None:
        start_index = int(xlim[0] * sample_rate)
        end_index = int(xlim[1] * sample_rate)
        data_slice = data[start_index:end_index]
        offset = xlim[0]
    else:
        data_slice = data
        offset = 0

    # Reshape data to (n_epochs, n_channels, n_times) as expected by MNE
    data_reshaped = data_slice[np.newaxis, np.newaxis, :]

    # Define a frequency vector from 1 Hz up to freq_max.
    num_freqs = int(np.floor(freq_max)) if freq_max > 1 else 1
    freqs = np.linspace(1, freq_max, num=num_freqs)

    # Compute the time-frequency representation using Morlet wavelets
    tfr = mne.time_frequency.tfr_array_morlet(data_reshaped, sfreq=sample_rate,
                                              freqs=freqs, n_cycles=n_cycles,
                                              decim=1)
    # Compute power as squared magnitude and convert to dB
    power = np.abs(tfr[0, 0])**2
    power_db = 10 * np.log10(power + 1e-10)

    # Create a time vector for the data_slice
    time = np.arange(data_slice.shape[0]) / sample_rate + offset

    # Plot the time-frequency representation
    im = ax.pcolormesh(time, freqs, power_db, shading='gouraud', cmap=cmap,
                       vmin=vmin, vmax=vmax)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(0, freq_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return im

# -----------------------------------------------------------------------
# Main script usage

# Replace with your file and group path
early_h5_filepath = '/Users/katehills/Library/CloudStorage/OneDrive-TheUniversityofManchester/PhD/Analysis/Chronic_Analysis/C01/AllTIDs/p2_h5/213/M1647066484_2022-03-12-06-28-04_tids_[213].h5'
early_hierarchy = 'M1647066484/213'
# Use the h5 filename (without extension) as a base for saving the figure
early_signal = os.path.splitext(os.path.basename(early_h5_filepath))[0]

# Load data
early_data = load_data_from_h5(early_h5_filepath, early_hierarchy, early_signal)

# Define the time window (in seconds) for zooming
selected_xlim = (2280, 2340)

# Use GridSpec to create:
fig = plt.figure(figsize=(18, 6.5))
gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 0.05], height_ratios=[1, 1],
                       wspace=0.05, hspace=0.3)

# Main axes for ECoG trace (top) and time-frequency plot (bottom)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

# Colorbar axis in the same row as the time-frequency plot
cax = fig.add_subplot(gs[1, 1])

# Set transparent backgrounds
fig.patch.set_alpha(0)
ax1.set_facecolor('none')
ax2.set_facecolor('none')

# --- Time Series Plot (top) ---
plot_time_series(ax1, early_data, sample_rate=256, color='k', linewidth=1)
ax1.set_xlim(selected_xlim)
ax1.set_ylim(-600, 600)
ax1.set_title('ECoG')

# --- Time-Frequency Plot (bottom) using MNE ---
im = plot_time_frequency(ax2, early_data, sample_rate=256, xlim=selected_xlim,
                         freq_max=120, n_cycles=7, cmap='viridis', vmin=30, vmax=60)

# Add the colorbar in the designated axis
cbar = fig.colorbar(im, cax=cax, label='Power/Frequency (dB/Hz)')

fig.suptitle('TID59_GRE', fontsize=16)

# Save the figure with a transparent background using the h5 filename as a base
png_filename = f"{early_signal}_seizure_{selected_xlim[0]}-{selected_xlim[1]}_k.png"
svg_filename = f"{early_signal}_seizure_{selected_xlim[0]}-{selected_xlim[1]}_k.svg"
plt.savefig(png_filename, transparent=True)
plt.savefig(svg_filename, transparent=True)

# Plot the figure; the figure will remain open until you close it manually.
#plt.show()
