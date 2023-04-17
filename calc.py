import scipy.signal as signal
from csaps import csaps
import numpy as np



#Correlation function and time-delay calculations
def lag_finder(y1, y2, sr, smooth_factor):
    n = len(y1)

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])
    corr_size = len(corr)
    corr_smooth = csaps(np.linspace(0, corr_size - 1, corr_size), corr, np.linspace(0, corr_size - 1, corr_size*smooth_factor), smooth=0.85)

    delay_arr = np.linspace(-0.5*n/(sr*smooth_factor), 0.5*n/(sr*smooth_factor), n * smooth_factor)
    delay = delay_arr[np.argmax(corr_smooth)]
    return delay_arr, corr_smooth, delay

def get_minmax_frequencies(spectrum, data):
    normalization = len(spectrum)/len(data)
    point = len(spectrum)//2
    abs_spectrum = np.abs(spectrum)
    max_indices = (point - abs_spectrum[:point].argmax(axis=0))/normalization

def get_spectrum(data, sample_period):
    num_of_samples = data.shape[0]  # returns shape of matrix
    time = num_of_samples*sample_period
    # Generate frequency axis and take FFT
    # Use FFT shift to get monotonically increasing frequency
    freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
    freq = np.fft.fftshift(freq)
    # takes FFT of all channels
    spectrum = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    return spectrum, freq, time

def get_axis(length, time, upscaling_factor):
    x = np.linspace(0, length - 1, length)
    x_new = np.linspace(0, length - 1, upscaling_factor)
    xsec = np.linspace(0, time, length)
    x_sec = np.linspace(0, time, upscaling_factor)
    return x, x_new, xsec, x_sec

def get_angle(delay1, delay2, delay3):
    theta = np.arctan(np.sqrt(3) * (delay1 + delay2) / (delay1 - delay2 - 2 * delay3))

    x = -(delay1 - delay2 - 2 * delay3)
    print("X equals: ", x)
    print("Unmodified Theta: ", theta / (2 * np.pi) * 360)
    
    if x < 0:
        theta += np.pi

    if theta < 0:
        theta += 2 * np.pi

    theta_degrees = theta / (2 * np.pi) * 360
    
    return theta, theta_degrees

def autocorrelate(sig, n_sample, smooth_factor):
    signal_length = len(sig)
    y = signal.correlate(sig, sig, "full")
    corr_length = len(y)
    corr_smooth = csaps(
        np.linspace(0, corr_length - 1, corr_length), 
        y, 
        np.linspace(0, corr_length - 1, corr_length*smooth_factor), 
        smooth=0.85
    )
    x = np.linspace(-0.5*signal_length/n_sample, 0.5*signal_length/n_sample, len(corr_smooth))

    return corr_smooth, x 

def moving_average(data, window):
    new_data = []
    data = np.append(data, np.zeros((window,), dtype=float))
    for index in range(0, len(data) - window):
        current_sum = 0
        for i in range (0, window):
            current_sum += data[i + index]
        current_avg = current_sum/window
        new_data.append(current_avg)
    return new_data

def moving_mean(data, window):
    return np.convolve(data, np.ones(window), 'valid') / window
