import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scp
from scipy.fft import rfft, rfftfreq
from calc import *

def execute_order_66(path):
    
    # Get data from file
    with open(path, 'r') as fid:
        data = np.loadtxt(fid)
        # print("Length of data is now: ", len(data), "\n Looks like: ", data, "\n\n")

    # Get red, green and blue channels from data
    c_r = data[:,0] # Channel: Red
    c_g = data[:,1] # Channel: Green
    c_b = data[:,2] # Channel: Blue

    # Set samplerate and duration from file
    fs = 40
    time = (len(c_r) + 2) / 40

    # Create a time axis to use for plotting etc
    t = np.linspace(0, time, int(time*fs-2))

    # Create filter
    f_lower_bound = 40
    f_upper_bound = 180
    b, a = scp.butter(5, [2*f_lower_bound/(fs*60), 2*f_upper_bound/(fs*60)], 'bandpass', analog=False)

    # Smarts
    m_avg_cr = moving_average(c_r, 10)
    new_cr = c_r - m_avg_cr

    m_mean_cr = moving_mean(c_r, 10)
    newer_cr = c_r[0:1189] - m_mean_cr[0:1189]

    # RED: Creating the bode plots and axis for the original and filtered signal
    spectrum_c_r = rfft(c_r)
    x_spectrum_cr = rfftfreq(len(c_r), 1/fs)
    x_spectrum_cr_bpm = x_spectrum_cr*60

    f_cr = scp.filtfilt(b, a, c_r)
    f_spectrum_cr = rfft(f_cr)
    x_f_spectrum_cr = rfftfreq(len(f_cr), 1/fs)
    x_f_spectrum_cr_bpm = x_f_spectrum_cr*60

    spectrum_cr_ma = rfft(new_cr)
    x_spectrum_cr_ma = rfftfreq(len(new_cr), 1/fs)
    x_spectrum_cr_ma_bpm = x_spectrum_cr_ma*60

    # GREEN: Creating the bode plots and axis for the original and filtered signal
    spectrum_c_g = rfft(c_g)
    x_spectrum_cg = rfftfreq(len(c_g), 1/fs)
    x_spectrum_cg_bpm = x_spectrum_cg*60

    f_cg = scp.filtfilt(b, a, c_g)
    f_spectrum_cg = rfft(f_cg)
    x_f_spectrum_cg = rfftfreq(len(f_cg), 1/fs)
    x_f_spectrum_cg_bpm = x_f_spectrum_cg*60

    # GREEN: Creating the bode plots and axis for the original and filtered signal
    spectrum_c_b = rfft(c_b)
    x_spectrum_cb = rfftfreq(len(c_b), 1/fs)
    x_spectrum_cb_bpm = x_spectrum_cb*60

    f_cb = scp.filtfilt(b, a, c_b)
    f_spectrum_cb = rfft(f_cb)
    x_f_spectrum_cb = rfftfreq(len(f_cb), 1/fs)
    x_f_spectrum_cb_bpm = x_f_spectrum_cb*60
    bpm = np.array(
        [round(x_f_spectrum_cr_bpm[np.argmax(np.abs(f_spectrum_cr))], 2), 
         round(x_f_spectrum_cg_bpm[np.argmax(np.abs(f_spectrum_cg))], 2), 
         round(x_f_spectrum_cb_bpm[np.argmax(np.abs(f_spectrum_cb))], 2)
         ]
        )

    #
    # All the plots below
    #
    '''
    # RED: Plot stuff 

    fig, axs = plt.subplots(2, 2)
    # Original signal
    axs[0, 0].set(title="Time domain signal", xlabel="Time [s]", ylabel="Signal amplitude")
    axs[0, 0].plot(t[0:1189], newer_cr, 'r', t, new_cr, 'b')

    # Bode of original signal
    axs[0, 1].set(title="Fourier transform", xlabel="Frequency [Bpm]", ylabel="Response [dB]")
    axs[0, 1].plot(x_spectrum_cr_ma_bpm[10:125], np.abs(spectrum_cr_ma[10:125]), "r")
    # axs[0, 1].set_yscale("log")
    # axs[0, 1].set_xlim(10, 250)

    # Filtered signal
    axs[1, 0].set(title="Time domain signal (Filter)", xlabel="Time [s]", ylabel="Signal amplitude")
    axs[1, 0].plot(t, f_cr, 'r')

    # Bode of filtered signal
    axs[1, 1].set(title="Fourier transform (Filter)", xlabel="Frequency [Bpm]", ylabel="Response [dB]")
    axs[1, 1].plot(x_f_spectrum_cr_bpm, np.abs(f_spectrum_cr), "r")
    # axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlim(10, 250) 

    # Adjustments and show/save plots
    plt.subplots_adjust(hspace=0.3, wspace=0.15, left=0.05, right=0.98, bottom=0.1, top=0.95)
    # plt.savefig("filename.png", dpi=300)
    plt.show()


    # GREEN: Plot stuff

    fig, axs = plt.subplots(2, 2)
    # Original signal
    axs[0, 0].set(title="Time domain signal", xlabel="Time [s]", ylabel="Signal amplitude")
    axs[0, 0].plot(t, c_g, 'g')

    # Bode of original signal
    axs[0, 1].set(title="Fourier transform", xlabel="Frequency [Bpm]", ylabel="Response [dB]")
    axs[0, 1].plot(x_spectrum_cg_bpm, np.abs(spectrum_c_g), "g")
    # axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlim(10, 250)

    # Filtered signal
    axs[1, 0].set(title="Time domain signal (Filter)", xlabel="Time [s]", ylabel="Signal amplitude")
    axs[1, 0].plot(t, f_cg, 'g')

    # Bode of filtered signal
    axs[1, 1].set(title="Fourier transform (Filter)", xlabel="Frequency [Bpm]", ylabel="Response [dB]")
    axs[1, 1].plot(x_f_spectrum_cg_bpm, np.abs(f_spectrum_cg), "g")
    # axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlim(10, 250)

    # Adjustments and show/save plots
    plt.subplots_adjust(hspace=0.3, wspace=0.15, left=0.05, right=0.98, bottom=0.1, top=0.95)
    # plt.savefig("filename.png", dpi=300)
    plt.show()


    # BLUE: Plot stuff

    fig, axs = plt.subplots(2, 2)
    # Original signal
    axs[0, 0].set(title="Time domain signal", xlabel="Time [s]", ylabel="Signal amplitude")
    axs[0, 0].plot(t, c_b, 'b')

    # Bode of original signal
    axs[0, 1].set(title="Fourier transform", xlabel="Frequency [Bpm]", ylabel="Response [dB]")
    axs[0, 1].plot(x_spectrum_cb_bpm, np.abs(spectrum_c_b), "b")
    # axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlim(10, 250)

    # Filtered signal
    axs[1, 0].set(title="Time domain signal (Filter)", xlabel="Time [s]", ylabel="Signal amplitude")
    axs[1, 0].plot(t, f_cb, 'b')

    # Bode of filtered signal
    axs[1, 1].set(title="Fourier transform (Filter)", xlabel="Frequency [Bpm]", ylabel="Response [dB]")
    axs[1, 1].plot(x_f_spectrum_cb_bpm, np.abs(f_spectrum_cb), "b")
    # axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlim(10, 250)

    # Adjustments and show/save plots
    plt.subplots_adjust(hspace=0.3, wspace=0.15, left=0.05, right=0.98, bottom=0.1, top=0.95)
    # plt.savefig("filename.png", dpi=300)
    plt.show()

    '''

    # compare: Plot stuff
    fig, axs = plt.subplots(2, 2)
    # Original signal
    axs[0, 0].set(title="Time domain signal", xlabel="Time [s]", ylabel="Signal amplitude")
    axs[0, 0].plot(t, c_r, 'r', t, c_g, 'g', t, c_b, 'b')

    # Bode of original signal
    axs[0, 1].set(title="Fourier transform", xlabel="Frequency [Bpm]", ylabel="Response")
    axs[0, 1].plot(
        x_spectrum_cr_bpm[10:len(x_spectrum_cb_bpm)], np.abs(spectrum_c_r[10:len(x_spectrum_cb_bpm)]), "r", 
        x_spectrum_cg_bpm[10:len(x_spectrum_cb_bpm)], np.abs(spectrum_c_g[10:len(x_spectrum_cb_bpm)]), "g", 
        x_spectrum_cb_bpm[10:len(x_spectrum_cb_bpm)], np.abs(spectrum_c_b[10:len(x_spectrum_cb_bpm)]), "b"
    )
    # axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlim(10, 250)

    # Filtered signal
    axs[1, 0].set(title="Time domain signal (Filter)", xlabel="Time [s]", ylabel="Signal amplitude")
    axs[1, 0].plot(t, f_cr, 'r', t, f_cg, 'g', t, f_cb, 'b')

    # Bode of filtered signal
    axs[1, 1].set(title="Fourier transform (Filter)", xlabel="Frequency [Bpm]", ylabel="Response")
    axs[1, 1].plot(x_f_spectrum_cr_bpm, np.abs(f_spectrum_cr), "r", x_f_spectrum_cg_bpm, np.abs(f_spectrum_cg), "g", x_f_spectrum_cb_bpm, np.abs(f_spectrum_cb), "b")
    # axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlim(10, 250)

    # Adjustments and show/save plots
    plt.subplots_adjust(hspace=0.3, wspace=0.15, left=0.05, right=0.98, bottom=0.1, top=0.95)
    # plt.savefig("filename.png", dpi=300)
    plt.show() 
    
    """ plt.title("Ufiltrerte signaler")
    plt.xlabel("Tid [S]")
    plt.ylabel("Kamera respons [0, 255]")
    plt.plot(t, c_r, 'r', label="Red")
    plt.plot(t, c_g, 'g', label="Green")
    plt.plot(t, c_b, 'b', label="Blue")
    plt.legend(loc=1)
    plt.savefig("r_raw_signal", dpi=300)
    plt.show()

    plt.title("Filtrerte signaler")
    plt.xlabel("Tid [S]")
    plt.ylabel("Kamera respons [0, 255]")
    plt.plot(t, f_cr, 'r', label="Red")
    plt.plot(t, f_cg, 'g', label="Green")
    plt.plot(t, f_cb, 'b', label="Blue")
    plt.legend(loc=1)
    plt.xlim(10, 20)
    plt.savefig("r_filtered_signal", dpi=300)
    plt.show()

    plt.title("Effektspektrum av filtrerte signaler")
    plt.xlabel("Frekvens [Bpm]")
    plt.ylabel("Relativ effekt [dB]")
    plt.plot(x_f_spectrum_cr_bpm[10:len(x_spectrum_cr_bpm)//6], np.abs(f_spectrum_cr[10:len(x_spectrum_cr_bpm)//6]), 'r', label="Red")
    plt.plot(x_f_spectrum_cg_bpm[10:len(x_spectrum_cg_bpm)//6], np.abs(f_spectrum_cg[10:len(x_spectrum_cg_bpm)//6]), 'g', label="Green")
    plt.plot(x_f_spectrum_cb_bpm[10:len(x_spectrum_cb_bpm)//6], np.abs(f_spectrum_cb[10:len(x_spectrum_cb_bpm)//6]), 'b', label="Blue")
    plt.legend(loc=1)
    plt.savefig("r_filtered_spectrum", dpi=300)
    plt.show() """

    print(path, bpm)
    return bpm

file_list_r = [
    "R_SR1.txt",
    "R_SR2.txt",
    "R_SR3.txt",
    "R_SR4.txt",
    "R_SR5.txt"
]

file_list_t = [
    "mt_1c.txt",
    "mt_2c.txt",
    "mt_3c.txt",
    "mt_4c.txt",
    "mt_5c.txt"
]

std_r_r = np.zeros(0)
std_r_g = np.zeros(0)
std_r_b = np.zeros(0)
std_t_r = np.zeros(0)
std_t_g = np.zeros(0)
std_t_b = np.zeros(0)
snitt_r = np.zeros(3)
snitt_t = np.zeros(3)
for i in range(0, len(file_list_r)):
   bpm_r = execute_order_66(file_list_r[i])
   # bpm_t = execute_order_66(file_list_t[i])
   std_r_r = np.append(std_r_r, bpm_r[0])
   std_r_g = np.append(std_r_g, bpm_r[1])
   std_r_b = np.append(std_r_b, bpm_r[2])
   """ std_t_r = np.append(std_t_r, bpm_t[0])
   std_t_g = np.append(std_t_g, bpm_t[1])
   std_t_b = np.append(std_t_b, bpm_t[2]) """

   for k in range(0,3):
       snitt_r[k] += bpm_r[k]
       # snitt_t[k] += bpm_t[k]

val_std_r_r = np.std(std_r_r)
val_std_r_g = np.std(std_r_g)
val_std_r_b = np.std(std_r_b)
val_std_t_r = np.std(std_t_r)
val_std_t_g = np.std(std_t_g)
val_std_t_b = np.std(std_t_b)

print("STD Reflektans R: ", val_std_r_r)
print("STD Reflektans G: ", val_std_r_g)
print("STD Reflektans B: ", val_std_r_b)
print("STD Transmittans R: ", val_std_t_r)
print("STD Transmittans G: ", val_std_t_g)
print("STD Transmittans B: ", val_std_t_b)

snitt_r = snitt_r/len(file_list_r)
snitt_t = snitt_t/len(file_list_t)
snitt_rgb_r = (snitt_r[0] + snitt_r[1] + snitt_r[2])/3
snitt_rgb_t = (snitt_t[0] + snitt_t[1] + snitt_t[2])/3


print("----------------------------------")
print("Reflektans: ")
print(snitt_r)
print(snitt_rgb_r)

print("----------------------------------")
print("Transmittans: ")
print(snitt_t)
print(snitt_rgb_t)


print(std_r_r)
print(std_r_g)
print(std_r_b)
print(std_t_r)
print(std_t_g)
print(std_t_b)