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
human_data = data[data['Model'] == 'Human'].drop(columns='Average')
human_means = data['Average'][-1:]
machine_data = data[data['Model'] != 'Human'].drop(columns='Average')
machine_means = data['Average'][:len(machine_data)]

# Calculate mean scores for each machine model
machine_data['Mean Score'] = machine_means
top_3_models = machine_data.nlargest(3, 'Mean Score')['Model'].values
machine_data.drop(columns=['Mean Score'], inplace=True)

# Data Transformation
melted_data = machine_data.melt(id_vars=['Model'], var_name='Category', value_name='Score')
filtered_positive_data = melted_data[melted_data['Score'] > 0]

# Map refined colors to the filtered data
filtered_positive_data['Color'] = filtered_positive_data['Category'].map(color_map)

# Top-3 models
highlight_models = list(top_3_models)

# Define distinct markers for the top-3 models and Human
refined_model_markers = {
    'o1-preview': {'marker': 'D', 'color': '#1F4E79'},          # Diamond, distinct dark blue
    'Claude-3.5-Sonnet': {'marker': 'P', 'color': '#357ABD'},   # Plus, medium blue
    'GPT-4o': {'marker': 's', 'color': '#3E70FF'}               # Square, distinct blue
}

# Calculate the mean scores for top-3 models and Human for each category
top_3_means = machine_data[machine_data['Model'].isin(top_3_models)].drop(columns=['Model']).mean()
human_mean = human_data.drop(columns=['Model']).mean()

# Plot refined box plot
plt.figure(figsize=(24, 10))  # Wider aspect ratio for a more professional layout
edge_palette = [color_map[cat] for cat in filtered_positive_data['Category'].unique()]
face_palette = [(plt.matplotlib.colors.to_rgba(color)[:3] + (0.8,)) for color in edge_palette]  # Add alpha transparency
box = sns.boxplot(
    data=filtered_positive_data,
    x='Category',
    y='Score',
    linewidth=1.5,
    zorder=1,
    patch_artist=True,
    width=0.7
)
for p, patch in enumerate(box.patches):
    patch.set_facecolor(face_palette[p])
    patch.set_edgecolor("gray")
    patch.set_linewidth(1.5)

# Overlay points for all machine models (excluding Human and top-3) with smaller markers
plt.scatter(
    x=np.tile(np.arange(len(top_3_means)), len(machine_data[~machine_data['Model'].isin(top_3_models)])),
    y=machine_data[~machine_data['Model'].isin(top_3_models)].drop(columns=['Model']).values.flatten(),
    label='Other LLMs',
    s=50,  # Smaller marker size
    marker='o',
    color=[color_map[cat] for cat in machine_data.drop(columns=['Model']).columns] * len(
        machine_data[~machine_data['Model'].isin(top_3_models)]),
    edgecolor="gray",
    alpha=0.7
)

# Overlay points for top 3 models and Human with refined markers and sizes
for model in highlight_models:
    model_config = refined_model_markers.get(model, {'marker': 'o', 'color': '#1E90FF'})  # Default marker and color
    marker = model_config['marker']
    color = model_config['color']

    model_scores = machine_data[machine_data['Model'] == model].drop(columns=['Model']).values.flatten()
    positive_scores = model_scores[(model_scores > 0) & (model_scores <= 100)]
    plt.scatter(
        x=np.arange(len(positive_scores)),
        y=positive_scores,
        label=model,
        s=150,
        marker=marker,
        edgecolors='black',
        color=color,
        alpha=0.9
    )

# Calculate overall average scores for all machine models
all_machine_mean = machine_data.drop(columns=['Model']).mean()

# Plot mean lines for all machine models, top-3 models, and Human
plt.plot(
    range(len(all_machine_mean)),
    all_machine_mean,
    label='All LLMs Avg',
    color='green',
    linestyle='--',
    linewidth=2.5
)
plt.plot(
    range(len(human_mean)),
    human_mean,
    label='Human',
    color='red',
    linestyle='--',
    linewidth=2.5
)

# Adjust plot aesthetics for a polished look
plt.ylim(0, 100)  # Limit y-axis to 0-100
plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='bold')
plt.tick_params(axis='x', direction='out', length=6)
plt.yticks(fontsize=14)
plt.tick_params(axis='y', direction='out', length=6)
plt.ylabel('Score', fontsize=18, fontweight='bold')  # Increase y-axis label font size
plt.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.7, color='gray')

handles, labels = plt.gca().get_legend_handles_labels()
new_handles = handles[1: 4] + handles[:1] + handles[4:]
new_labels = labels[1: 4] + labels[:1] + labels[4:]
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
plt.savefig(os.path.join(save_dir, "subcategory_v0.8.pdf"), format='png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
plt.close()
