import solve_lab_2
import time
import numpy as np


filename_list = [
    "sine_far_5.bin", 
    "sine_far_1.bin", 
    "sine_far_2.bin", 
    "sine_far_3.bin", 
    "sine_far_4.bin",
]

plot_list = [True, False, False, False, False]

samplerate = 31250
offset = 1.65
lower_bound = 10
upper_bound = 1
smooth_factor = 8

dataset = []
for i in range(0, len(filename_list)):
    angle = solve_lab_2.main(filename_list[i], plot_list[i], offset, smooth_factor, lower_bound, upper_bound, samplerate)
    dataset.append(angle)
    time.sleep(1)
print(dataset)
print(round(np.std(dataset), 5))