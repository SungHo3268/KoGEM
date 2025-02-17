import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


kogem_dataset = json.load(open("datasets/KoGEM_benchmark.json", "r"))
s1_df = pd.read_excel("s1_elapsed_time.xlsx")

kogem_info = json.load(open("utils/KoGEM_info.json", "r"))
major_categories = kogem_info['major_categories']
sub_categories = kogem_info['sub_categories']

major_time = {key: [] for key in major_categories}
sub_time = {key: [] for key in sub_categories}
major_score = {key: [] for key in major_categories}
sub_score = {key: [] for key in sub_categories}
for i in range(len(s1_df)):
    data = s1_df.iloc[i]
    kogem_data = kogem_dataset[data.idx]
    if str(data.time) == 'nan':
        continue
    else:
        elapsed_time = int(data.time.split()[0][:-1]) * 60 + int(data.time.split()[1][:-1])
        major_time[data.level_1].append(elapsed_time)
        sub_time[data.level_2].append(elapsed_time)

        major_score[data.level_1].append(data.correct)
        sub_score[data.level_2].append(data.correct)

# To plot the graph, set the average data

sub_avg_times = {key: np.mean(sub_time[key]) for key in sub_time}
# sub_med_times = {key: np.median(sub_time[key]) for key in sub_time}

# sub_avg_score = {key: np.mean(sub_score[key])*100 for key in sub_score}
sub_scores = [53.85, 37.97, 40.23, 44.83, 44.74, 44.14, 50.00, 65.87, 61.82, 54.72, 44.74, 33.33, 33.33, 33.33, 50.00, 14.29]
sub_avg_score = {key: sub_scores[i] for i, key in enumerate(sub_score)}

# Extract keys and values for plotting
subcategories = list(sub_avg_times.keys())[::-1]
avg_times = list(sub_avg_times.values())[::-1]
# med_times = list(sub_med_times.values())[::-1]
sub_avg_score = list(sub_avg_score.values())[::-1]

# Get the top-3 longest elapsed times
top_3_times = sorted(avg_times, reverse=True)[:3]

# Create the figure and axis for the plot
x_len = 9
fig, ax1 = plt.subplots(figsize=(x_len, 8))

# Plot the histogram
ax1.barh(subcategories, avg_times, color='#E4C9AE', alpha=0.7, label='mean time')
# ax1.plot(med_times, subcategories, color='#CD4F41', marker='o', linewidth=2.5, label='median time')

# Add a secondary x-axis for average scores
ax2 = ax1.twiny()
# ax2.plot(sub_avg_score, subcategories, color='#1C404C', marker='s', linestyle='-', linewidth=2.5, label='average score')
# ax2.plot(sub_avg_score, subcategories, color='#CD4F41', marker='s', linestyle='-', linewidth=3, label='average score')

# Set labels and limits for the axes
ax1.set_xlabel('Average Elapsed Time (s)', fontsize=18, color='black')
ax2.set_xlabel('Average Score (%)', fontsize=18, color='black')
ax1.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='x', labelsize=14)
ax1.yaxis.set_tick_params(labelsize=18)
if x_len == 10:
    ax1.set_xlim(0, 290)
elif x_len == 9:
    ax1.set_xlim(0, 290)
elif x_len == 8:
    ax1.set_xlim(0, 320)
else:
    raise ValueError

ax2.set_xlim(0, 100)  # Assuming scores range between 0 and 1

# Add dashed vertical lines for the top-3 longest elapsed times
for t in top_3_times:
    ax1.axvline(x=t, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)

# Show the plot
plt.tight_layout()

# Enhance the boundary visibility
for spine in ax2.spines.values():
    spine.set_linewidth(1)  # Set thicker line for primary axis
    spine.set_edgecolor('lightgrey')  # Set boundary color to black

ax1.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)  # Remove the right y-axis boundary

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax1.legend(lines_1[::-1] + lines_2, labels_1[::-1] + labels_2, fontsize=14, loc='upper right', bbox_to_anchor=(1.05, .32), facecolor='#f5f5f5')  # A brighter shade of light grey
plt.savefig("analysis/assets/figures/elapsed_time.pdf")
plt.show()
