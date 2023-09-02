import matplotlib.pyplot as plt
import numpy as np
import os
def plot_feature(folder, feature, start, end, data, data_filtered, channel_name, hfo_start, hfo_end):
    channel_data = data
    channel_data_f = data_filtered
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    channel_data = np.squeeze(channel_data)
    channel_data_f = np.squeeze(channel_data_f)
    ax1.imshow(feature[0])
    ax2.plot(channel_data, color='blue')
    ax2.plot(np.arange(hfo_start, hfo_end), channel_data[hfo_start:hfo_end], color='red')
    ax3.plot(channel_data_f, color='blue')
    ax3.plot(np.arange(hfo_start, hfo_end), channel_data_f[hfo_start:hfo_end], color='red')
    plt.suptitle(f"{channel_name}_{start}_{end} with length: {(end - start)*0.5} ms")
    fn = f'{channel_name}_{start}_{end}.jpg'
    plt.savefig(os.path.join(folder,fn))
    plt.close()