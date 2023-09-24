import pandas as pd
import matplotlib.pyplot as plt
import os

font_size = 30  # Define a common font size variable

# Set the font to Arial
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"

# Define the paths to the CSV files
unzip_dir = './'
files = ['loss_tau_0.04.csv', 'loss_tau_0.02.csv', 'loss_tau_0.01.csv', 'loss_odeint.csv']

# Load the CSV files into dataframes
df_tau_004 = pd.read_csv(os.path.join(unzip_dir, files[0]))
df_tau_002 = pd.read_csv(os.path.join(unzip_dir, files[1]))
df_tau_001 = pd.read_csv(os.path.join(unzip_dir, files[2]))
df_odeint = pd.read_csv(os.path.join(unzip_dir, files[3]))

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the loss curves
ax.plot(df_tau_001['Step'], df_tau_001['Value'], label='Euler Method (0.01)', linewidth=2, ls='-.')
ax.plot(df_tau_002['Step'], df_tau_002['Value'], label='Euler Method (0.02)', linewidth=2, ls='-.')
ax.plot(df_tau_004['Step'], df_tau_004['Value'], label='Euler Method (0.04)', linewidth=2, ls='-.')
ax.plot(df_odeint['Step'], df_odeint['Value'], label='RK45', linewidth=2)

# Set axes labels
ax.set_xlabel('Thousand Iteration', fontsize=font_size)
ax.set_ylabel('Accumulated  Cost', fontsize=font_size)
ax.set_yscale("log")

# Adjust x-ticks to be in terms of 10^3 (thousand iterations)
ticks = ax.get_xticks()
ax.set_xticks(ticks)
ax.set_xticklabels([str(tick/1e3) for tick in ticks], fontsize=font_size)
ax.tick_params(axis='y', labelsize=font_size)

plt.xlim(0, 30000)
# Add grid and legend
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
ax.legend(fontsize=font_size)

# Show the plot (you can comment this out if you only want to save the plot)
plt.tight_layout()

# Save the figure as a PDF
plt.savefig('model_based_motivation.pdf', bbox_inches='tight')
plt.show()
