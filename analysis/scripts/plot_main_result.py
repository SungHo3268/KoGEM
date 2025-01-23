"""
This file yields the Figure 1 in KoGEM paper
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


kogem_info = json.load(open('utils/KoGEM_info.json', "r"))
categories = kogem_info["major_categories"]
num_vars = len(categories)

# Data preperation
data = pd.read_excel("analysis/assets/exp_results/linguistic_category_results.xlsx", sheet_name='Sheet1')

results = {}
for i in range(len(data)):
    cur_data = data.loc[i]
    model = cur_data['Model']
    results[model] = []
    for category in categories:
        results[model].append(cur_data[category])


# Sort results dictionary
sorted_models = ['o1-preview', 'Human', 'Claude-3.5-Sonnet', 'GPT-4o']
results = {model: results[model] for model in sorted_models if model in results}


fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
ax.set_facecolor("#f2f2f5")  # Set seaborn's default background color
ax.set_theta_zero_location("N")         # N for North
ax.set_theta_direction(-1)              # -1 for counterclockwise


# plot each model
start_point = 2/5 * np.pi
angles = np.linspace(start_point, start_point + 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]        # for closing the circle

# Specify colors manually
colors = ["#1F78B4",  # Richer Blue
          "#E31A1C",  # Richer Red
          "#33A02C",  # Richer Green
          "#FF7F00",  # Richer Orange
          ]
alphas = [0.1, 0.15, 0.1, 0.1]

for (model_name, values), color, alpha in zip(results.items(), colors, alphas):
    values += values[:1]    # for closing the circle
    if model_name == 'Human':
        ax.plot(angles, values, 'o-', label=model_name, linewidth=2.5, color=color, markersize=11)
    else:
        ax.plot(angles, values, 'o-', label=model_name, linewidth=2, color=color, markersize=11)
    ax.fill(angles, values, alpha=alpha, color=color)


# Set xlabels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=18,
                   )

# Set ylabels
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=14, color="black")
ax.set_ylim(20, 100)

# Set polar axis visible
ax.spines["polar"].set_visible(False)
ax.grid(color="gray", linestyle="--", linewidth=1.0)


# Set the legends
handles, labels = plt.gca().get_legend_handles_labels()
new_handles = handles[: 1] + handles[2:] + handles[1:2]
new_labels = labels[: 1] + labels[2:] + labels[1:2]
plt.legend(
    fontsize=12,
    loc='upper right',
    bbox_to_anchor=(1.2, 1.2),  # Move legend to the right outside the plot
    fancybox=True,
    shadow=True,
    frameon=True,
    handles=new_handles,
    labels=new_labels,
)

# 차트 표시
plt.tight_layout()
# plt.savefig("analysis/assets/figures/main_v0.3.pdf")
plt.show()
