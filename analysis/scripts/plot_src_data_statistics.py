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
for data in dataset:
    src = data["data_src"]
    level_1 = data['level_1']
    level_2 = data['level_2']

    data_src_major[src][level_1] += 1

# Calculate total statistics for all categories
data_src_major['Total'] = {cat: sum(data_src_major[src][cat] for src in data_srcs) for cat in major_categories}


####################################
#  For each exam source (Pie Chart)
####################################
data_src_major_ratio = {
    key: {cat: round(data_src_major[key][cat] / sum(data_src_major[key].values()) * 100, 1) for cat in major_categories}
    for key in data_src_major}

fig, axes = plt.subplots(2, 5, figsize=(25, 10))  # Create a 2x5 grid for subplots

# Define the order of circles for plotting
circle_order = ["NUAT(HS1)", "NUAT(HS1)", "NUAT(HS1)", "CSAT", "HSQE", "LCSE(G9)", "LCSE(G7)", "NCSE(G9)", "NCSE(G7)", "Total"]

for ax, exam in zip(axes.flatten(), circle_order):
    data = data_src_major_ratio.get(exam, {})
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
