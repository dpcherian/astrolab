#!/usr/bin/env python

"""
This package contains a list of functions that can be used to analyse signals for the Doppler Effect experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def load_signal(filename, clip=None, print_log=False, color='royalblue', fig=None, ax=None):
    """
    Load a ``.wav`` file and return both the time-series information and the sample rate.

    Parameters
    ----------
    filename: str
        The ``.wav`` file to read: a filename pointing to the location of the file.

    clip: array_like, default: None
        A two-element list ``[a,b]`` and clips the input signal between the times ``a`` and ``b`` (in seconds).
        
        This can be useful if you want to remove the first ``a`` seconds or the last ``b`` seconds of your sound file. By default, no clipping is done.

    print_log: bool, default: False
        A boolean variable to decide whether you want to print a log of what you've done. In this case, whether you want to plot the resulting signal or not. By default, the signal is not plotted.

    color: str, default: "royalblue"
        Colour of this plot. Must be one of the matplotlib "named colors": https://matplotlib.org/stable/gallery/color/named_colors.html.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.

    Returns
    -------
    samplerate: int
        The sample rate of the signal. This is defined as the number of datapoints of the input signal per second. Typically, most recordings are taken at 44,100 Hz, meaning that there are 44,100 samples in each second.

    times: array_like
        An array of the time-steps at which the data is sampled, in seconds.

    signal: array_like
        An array containing the value of the amplitude of the signal at each time-step in times.

    Raises
    ------
    ValueError
        In case the ``clip`` array's entries are not in ascending order.

    Usage
    -----
    >>> sr, times, sig = load_signal("./filename.wav", clip=[3,20], print_log=True, color='firebrick', fig=None, ax=None)
    """
    samplerate, signal = wavfile.read(filename)  # Read the data into variables
    
    if(signal.ndim > 1):                         # If the audio is stereo, multiple channels exist
        signal = np.mean(signal, axis=1)         # Averaged over multiple channels to get a single array.
    
    if(print_log and ax is None):
        fig, ax = plt.subplots()                 # If no axis element is provided, create an empty axis.
        
    T = len(signal)/samplerate                   # Total signal time T
    times = np.linspace(0,T,len(signal))         # Time-steps (in seconds) of the signal
    
    if(clip is not None):                        # If clipping is to be done,
        if(clip[0] > clip[1]):                   # check if the clip array is arranged in ascending order,
            raise ValueError("Clipping Error: the `clip` array should be in ascending order, i.e. the code requires clip[0]<clip[1].")
        mask = (times>clip[0]) & (times<clip[1]) # create a mask of times to clip
        
        times = times[mask]                      # clip the arrays using this mask
        times = times-times[0]
        signal = signal[mask]
    
    if(print_log):                               # If a log must be printed,
        ax.plot(times, signal, color=color)      # plot the signal, with the specified colour
    
    return samplerate, times, signal             


def chunk_signal(times, signal, size=2048, step=128):
    """
    Break up a signal (given in the ``signal`` array) into individual pieces
    (or chunks) of size ``size``, whose left-edges are separated by a ``step`` samples. 
    
    Thus, if "chunk 0" ranges from (0, ``size``), "chunk 1" would range from (``step``, ``size + step``). Notice how both chunks are ``size`` samples wide, and their left edges are separated by ``step`` samples. The splitting is thus done with significant overlap. 
    
    For each of these chunks, it returns the average time at which the chunk was taken, and the signal's amplitude values within that chunk.

    Parameters
    ----------
    times: array_like
        An array of times, usually one of the outputs of the ``load_signal`` function.
    
    signal: array_like
        An array of amplitudes for each of the times in ``times``, also typically one of the outputs of the ``load_signal`` function. 
        
    size: int, default: 2048
        The number of samples in a single chunk. 
    
    step: int, default: 128 
        The number of samples between the left-edges of successive chunks. 

    Returns
    -------
    avg_t: array_like
        A one-dimensional array containing the average time values for each of the chunks.

    chunks: array_like
        A two-dimensional array. The first dimension is the chunk-number. For each chunk, it returns size number of amplitude values.

    Usage
    -----
    >>> avg_t, chunks = chunk signal(times, sig)
    """
    size = int(size); step = int(step)             # Cast size and step into integers, if necessary

    n_steps = len(signal)                          # Number of steps in the signal
    n_chunks = (n_steps-size) // step              # Number of chunks in the signal
    
    array = np.zeros((n_chunks, size))             # Empty 2-dimensional array to hold the chunks
    t = np.zeros(n_chunks)                         # Empty 1-dimensional array for the time-steps
    
    counter = 0                                    # Integer counter to count each chunk
    while(counter < n_chunks):                     # Until the total number of chunks is reached,
        start = counter*step                       # Choose the start-location of a chunk,
        end = start + size                         # and the end-location of a chunk.
        
        array[counter] = signal[start:end]         # Clip the signal between ``start`` and ``end``
        t[counter] = (times[start]+times[end])/2   # Compute the average time in the interval
        
        counter += 1                               # Increment the counter
    
    return t, array


def power_spectrum(chunk, samplerate, lowx=1000, highx=10_000, print_log=False, color='tomato', label=None, fig=None, ax=None):
    """
    Accept an array of signal values, compute its power spectrum, and return the dominant
    frequency in the range ``[lowx, highx]``. Ideally, this range should be centered around the expected "stationary" frequency, and should be wide enough contain all the variation in the frequency during the object's motion.

    Parameters
    ----------
    chunk: array_like
        An array of amplitude values, usually a specific chunk that was returned by the ``chunk_signal`` function, although the entire signal could be used as well.

    samplerate: int
        The original samplerate of the signal, obtained from the ``load_signal`` function.

    lowx: float, default: 1000
        Lower limit (in Hz) of the frequency-range within which the dominant frequency is found and (if ``print_log=True``) the result is plotted.
    
    highx: float, default: 10_000
        Upper limit (in Hz) of the frequency-range within which the dominant frequency is found and (if ``print_log=True``) the result is plotted.

    print_log: bool, default: False
        A boolean variable to decide whether you want to print a log of what you've done. In this case, if ``print_log=True``, the resulting power spectrum is plotted. By default, the power spectrum is not plotted.
        
    color: str, default: "tomato"
        Set the colour of the plot that is produced if ``print_log=True``.
    
    label: str or None, default: None
        Set the label of the plot that is produced if ``print_log=True``. By default, no label is set. 

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created. 

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.

    Returns
    -------
    maxfreq: The frequency in the range ``[lowx , highx]`` that contributes the greatest deal to our signal in the given range of frequencies.

    Usage:
    ------
    >>> dominant_frequency = power_spectrum(chunks[0], samplerate, lowx=2000, highx=4000)
    """
    signal = chunk - np.mean(chunk)           # Subtract the mean from the signal, so that it is oscillating about zero. This is to remove any overall "offset" to the signal.
    
    signal = signal*np.hamming(len(signal))   # The signal is multiplied by a Hamming window (a weighted cosine) to remove discontinuities at the edge, and reduce the edge effects due to the finite size of the signal.
        
    N = len(signal)

    powspec = np.abs(np.fft.fft(signal))      # Compute the power spectrum of the signal
    fftfreqs = np.fft.fftfreq(len(signal), d=1/samplerate) # Compute the range of frequencies of the power spectrum
    
    mask = (fftfreqs>lowx) & (fftfreqs<highx) # Mask of frequencies between the high and low values
    clipped_freqs = fftfreqs[mask]            # Apply the mask to the frequencies and 
    clipped_powspec = powspec[mask]           # the power spectrum
    
    if(print_log==True):                      # If a log must be printed,
        if(ax is None):
            fig, ax = plt.subplots()          # create an axis if one isn't already provided
        ax.plot(clipped_freqs, clipped_powspec, color=color, label= label) # Plot the power spectrum
    
    max_ps = np.max(clipped_powspec)          # Find the maximum of the power spectrum
    
    dom_freq = clipped_freqs[clipped_powspec==max_ps][0] # Find the dominant frequency in the range
    
    return dom_freq