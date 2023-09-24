import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

font_size = 30  # Define a common font size variable

# Set the font to Arial
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"

fig, ax = plt.subplots(figsize=(9, 6))

# Load data (replace these lines with your actual data loading)
# For demonstration, I'm using random data here.
J_seq_02_Matern = np.load('Matern_02.npz.npy')
J_seq_03_Matern = np.load('Matern_03.npz.npy')
J_seq_04_Matern = np.load('Matern_04.npz.npy')
# J_seq_05_Matern = np.load('Matern_05.npz.npy') * 100

J_seq_02_trapz = np.load('trapz_02.npz.npy')
J_seq_03_trapz = np.load('trapz_03.npz.npy')
J_seq_04_trapz = np.load('trapz_04.npz.npy')
# J_seq_05_trapz = np.load('trapz_05.npz.npy')

# Define labels for N values and the corresponding data sets
N_labels = ['N=2', 'N=3', 'N=4']
matern_data = [J_seq_02_Matern, J_seq_03_Matern, J_seq_04_Matern]
trapz_data = [J_seq_02_trapz, J_seq_03_trapz, J_seq_04_trapz]

# Create empty lists to store line objects
matern_lines = []
trapz_lines = []

# Plot using default color cycle for N values
for i, (matern, trapz) in enumerate(zip(matern_data, trapz_data)):
    line1, = ax.plot(matern, linestyle='solid', linewidth=3, alpha=0.8)
    matern_lines.append(line1)
    ax.scatter(range(len(matern)), matern, marker='P', s=100)

    line2, = ax.plot(trapz, linestyle='dotted', linewidth=3, alpha=0.8)
    trapz_lines.append(line2)
    ax.scatter(range(len(trapz)), trapz, marker='P', s=100)

ax.set_yscale('log')
# ax.set_ylim(0, 1000000)

# Customize the appearance of the plot
ax.set_xlabel('Iteration', fontsize=font_size)
ax.set_ylabel(r'$|J-J^*|$', fontsize=font_size)
ax.tick_params(axis='x', labelsize=font_size)
ax.tick_params(axis='y', labelsize=font_size)

# Create the custom legend
custom_handles = []
custom_labels = []

for i, label in enumerate(N_labels):
    custom_handles.append((matern_lines[i], trapz_lines[i]))
    custom_labels.append(f'{label} (Matèrn, Trapz)')

ax.legend(custom_handles, custom_labels, loc='best', fontsize=font_size, ncol=1,
          handler_map={tuple: HandlerTuple(ndivide=None)})

plt.tight_layout()
plt.savefig(f'./sin_motivation.pdf')

# Show the plot
plt.show()
