import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter, FuncFormatter

font_size = 30  # Define a common font size variable

# Set the font to Arial
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"


# Custom formatter function
def my_formatter(x, pos):
    return "{:.0f}".format(x * 1e5)


folder_name = "./"

file_list = os.listdir(folder_name)
file_list = [f for f in file_list if f.endswith('.npy')]
file_list.sort()  

colormap = cm.get_cmap("tab10", len(file_list))

colors = [colormap(i) for i in range(len(file_list))]

fig, ax = plt.subplots(figsize=(9, 6))

w_seq = []
sample_points_squared = []
point_seq = []

for file, color in zip(file_list, colors):
    w, _, _ = np.load(os.path.join(folder_name, file), allow_pickle=True)
    sample_point = int(file.split("_")[1].split(".")[0])

    w_seq.append(w)
    sample_points_squared.append(1.0 / (sample_point ** 2))  # Storing the square of sample_point
    point_seq.append(sample_point)

w_seq = np.array(w_seq)
sample_points_squared = np.array(sample_points_squared)
point_seq = np.array(point_seq)

ax.plot(sample_points_squared, w_seq, color='ForestGreen', linestyle='-', linewidth=2, alpha=0.8)
ax.scatter(sample_points_squared, w_seq, color='ForestGreen', marker='P', s=100)

ax.set_xlabel(r'${1}/{N^2}$', fontsize=font_size)
ax.set_ylabel(r'$||\hat{\omega}^{(\infty)}-\omega^*||_2$', fontsize=font_size)
ax.tick_params(axis='x', labelsize=font_size)
ax.tick_params(axis='y', labelsize=font_size)

yticks_original = np.array([3, 6, 9, 12]) * 1e-5
ax.set_yticks(yticks_original)
# Using FuncFormatter to format y-axis ticks
ax.yaxis.set_major_formatter(FuncFormatter(my_formatter))
ax.annotate(r'$\times 10^{-5}$',
            xy=(0.1, 1),
            xycoords='axes fraction',
            xytext=(0, 10),
            textcoords='offset points',
            fontsize=font_size,
            ha='right',
            va='bottom')

plt.tight_layout()
plt.savefig(f'./sin_trapz_w.pdf')
