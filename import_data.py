from csaps import csaps
import numpy as np


def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))
    return sample_period, data


def import_from_file(fname):
    try:
        sample_period, data = raspi_import(fname)
        data *= 0.0008
        sample_period *= 1e-6
    except FileNotFoundError as err:
        print(f"File {fname} not found. Check the path and try again.")
        exit(1)
    return sample_period, data

def get_channels(data, offset, sliced_lower, sliced_upper, x, x_new):
    channel1 = data[:,0] - offset
    channel2 = data[:,1] - offset
    channel3 = data[:,2] - offset

    #Raw data where the first couple of samples (and last) are cut out
    sliced1 = np.array(channel1[sliced_lower:sliced_upper])
    sliced2 = np.array(channel2[sliced_lower:sliced_upper])
    sliced3 = np.array(channel3[sliced_lower:sliced_upper])

    #Smoothed curves of input data
    sliced1_smooth = csaps(x, sliced1, x_new, smooth=0.85)
    sliced2_smooth = csaps(x, sliced2, x_new, smooth=0.85)
    sliced3_smooth = csaps(x, sliced3, x_new, smooth=0.85)
    return sliced1_smooth, sliced2_smooth, sliced3_smooth