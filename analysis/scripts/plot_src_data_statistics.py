import os
import json
from matplotlib import pyplot as plt


############################################
#     Load the dataset and its settings
############################################
dataset = json.load(open("datasets/KoGEM_benchmark.json", "r"))

KoGEM_info = json.load(open("utils/KoGEM_info.json", "r"))
major_categories = KoGEM_info['major_categories']
sub_categories = KoGEM_info['sub_categories']
data_srcs = KoGEM_info['data_srcs']
color_map = json.load(open("utils/color_map.json", "r"))

# Count the number of data for each category
data_src_major = {src_data: {cat: 0 for cat in major_categories} for src_data in data_srcs}
data_src_sub = {src_data: {cat: 0 for cat in sub_categories} for src_data in data_srcs}
for data in dataset:
    src = data["data_src"]
    level_1 = data['level_1']
    level_2 = data['level_2']

    data_src_major[src][level_1] += 1
    data_src_sub[src][level_2] += 1

"""
############################################
#   Plot Major Categories per Data Source
############################################
x = np.arange(len(major_categories))
width = 0.1  # the width of the bars
multiplier = 0
fig, ax = plt.subplots(layout='constrained', figsize=(30, 10))
for i, key in enumerate(data_src_major):
    offset = width * (multiplier + 0.2)  # Add extra spacing between groups
    cur_large_cat = data_src_major[key]
    rects = ax.bar(x=x + offset,
                   height=list(cur_large_cat.values()),
                   width=width,
                   color=[color_map[cat] for cat in major_categories],
                   edgecolor='gray',  # Add border to the bars
                   alpha=0.7,
                   label=key)
    ax.bar_label(rects, padding=3, fontsize=20)
    multiplier += 1
ax.set_xticks(x + width * (len(major_categories) // 2 + 1), labels=major_categories, fontsize=20)
ax.legend(loc='upper right', fontsize=16, ncols=2)
plt.show()


############################################
#   Plot SubCategories per Data Source
############################################
x = np.arange(len(sub_categories))
width = 0.1  # the width of the bars
multiplier = 0
fig, ax = plt.subplots(layout='constrained', figsize=(50, 10))
for i, key in enumerate(data_src_sub):
    offset = width * (multiplier + 0.2)  # Add extra spacing between groups
    cur_sub_cat = data_src_sub[key]
    rects = ax.bar(x=x + offset,
                   height=list(cur_sub_cat.values()),
                   width=width,
                   color=[color_map[cat] for cat in sub_categories],
                   edgecolor='gray',  # Add border to the bars
                   alpha=0.7,
                   label=key)
    ax.bar_label(rects, padding=3, fontsize=20)
    multiplier += 1
ax.set_xticks(x + width * (len(sub_categories) // 2 + 1), labels=sub_categories, fontsize=20)
ax.legend(loc='upper right', fontsize=16, ncols=2)
plt.show()
"""

####################################
#  For each exam source (Pie Chart)
####################################
data_src_major_ratio = {
    key: {cat: round(data_src_major[key][cat] / sum(data_src_major[key].values()) * 100, 1) for cat in major_categories}
    for key in data_src_major}
# data_src_sub_ratio = {key: {cat: round(data_src_sub[key][cat]/sum(data_src_sub[key].values())*100, 1) for cat in sub_categories} for key in data_src_sub}

fig, axes = plt.subplots(3, 3, figsize=(20, 20))  # Create a 3x3 grid for subplots

for ax, (exam, data) in zip(axes.flatten(), data_src_major_ratio.items()):
    sizes = [val for val in data.values() if val > 0]
    labels = [key for key, val in data.items() if val > 0]
    colors = [color_map[label] for label in labels]
    wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, colors=colors,
                                      textprops={'fontsize': 20, 'fontweight': 'bold', 'color': 'white'},
                                      pctdistance=0.7)

    for autotext, count in zip(autotexts, [data_src_major[exam][label] for label in labels]):
        autotext.set_text(f'{count}')
        autotext.set_fontsize(24)

    ax.set_title(exam, fontsize=20, weight='bold', pad=1)  # Reduce spacing with 'pad' parameter

plt.tight_layout()  # Adjust overall layout to prevent overlap

save_dir = "analysis/assets/figures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# plt.savefig(os.path.join(save_dir, f"data_src_statistics.pdf"))
plt.show()
