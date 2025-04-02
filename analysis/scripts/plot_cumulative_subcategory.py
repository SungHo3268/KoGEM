"""
This file yields the Figure 4 in KoGEM paper
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid", {"grid.color": "gray", "grid.linewidth": 1.5})

# Load basic information for KoGEM benchmark
kogem_info = json.load(open("utils/KoGEM_info.json", "r"))
color_map = json.load(open("utils/color_map.json", "r"))

# Data preparation
data = pd.read_excel("analysis/assets/exp_results/subcategory_results.xlsx", sheet_name='Sheet1')

# Extract human data and filter machine models
# human_data = data[data['Model'] == 'Human'].drop(columns='Average')
# human_means = data['Average'][-1:]
machine_data = data[data['Model'] != 'Human'].drop(columns='Average')
machine_means = data['Average'][:len(machine_data)]

# Calculate mean scores for each machine model
machine_data['Mean Score'] = machine_means
machine_data.drop(columns=['Mean Score'], inplace=True)


plt.figure(figsize=(16, 8))

# human_mean = human_data.drop(columns=['Model']).mean()
# plt.plot(
#     range(len(human_mean)),
#     human_mean,
#     label='Human',
#     color='red',
#     linestyle='-',
#     linewidth=3.0
# )

all_machine_mean = machine_data.drop(columns=['Model']).mean()
plt.plot(
    range(len(all_machine_mean)),
    all_machine_mean,
    label='All LLMs Avg',
    color="red",
    linestyle='-',
    linewidth=3.0
)

colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B"]
for idx, i in enumerate([1, 3, 5, 10, 15, 20]):
    # Calculate mean scores for models top-1, top-3, top-5, ..., top-20
    mean_top = machine_data.iloc[-i:].drop(columns=['Model']).mean()  # Get top-i models
    plt.plot(
        range(len(mean_top)),
        mean_top,
        label=f'Top-{i}' if i == 1 else f'Top-{i} Avg',
        color=colors[idx],
        linestyle='--',
        linewidth=3.0
    )

# Adjust plot aesthetics for a polished look
plt.ylim(15, 100)  # Limit y-axis to 0-100
plt.xticks(range(len(machine_data.columns) - 1), machine_data.columns[1:], rotation=45, ha='right', fontsize=14,
           fontweight='bold')
plt.tick_params(axis='x', direction='out', length=6)
plt.yticks(fontsize=14)
plt.tick_params(axis='y', direction='out', length=6)
plt.ylabel('Accuracy Score', fontsize=18, fontweight='bold')  # Increase y-axis label font size
plt.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.7, color='gray')


handles, labels = plt.gca().get_legend_handles_labels()
new_handles = handles[1:] + handles[:1]
new_labels = labels[1:] + labels[:1]
plt.legend(
    fontsize=12,
    loc='center left',
    bbox_to_anchor=(1.05, 0.5),  # Move legend to the right outside the plot
    fancybox=True,
    shadow=True,
    frameon=True,
    handles=new_handles,
    labels=new_labels,
)

plt.xlabel(None)  # Remove x-axis label

# Add a polished frame and adjust layout
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')

# Save the figure
save_dir = "analysis/assets/figures/"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "subcategory_cumulative.png"), format='png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

plt.close()
