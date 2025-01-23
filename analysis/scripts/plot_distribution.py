import json
import matplotlib.pyplot as plt


kogem_info = json.load(open("utils/KoGEM_info.json", "r"))

inner_labels = kogem_info["major_categories"]
inner_sizes = [213, 268, 438, 385, 220]     # (Phonology, Morphology, Syntax, Semantics, Norms)

outer_labels = kogem_info["sub_categories"]
outer_sizes = [
    26, 187,                    # (Phonology)
    29, 87, 152,                # (Morphology)
    290, 148,                   # (Syntax)
    167, 165, 53,               # (Semantics)
    114, 27, 18, 18, 36, 7,     # (Norms)
]

color_map = json.load(open("utils/color_map.json", "r"))

# Set Colors
inner_colors = [color_map[cat] for cat in inner_labels]
outer_colors = [color_map[cat] for cat in outer_labels]


fig, ax = plt.subplots(figsize=(8, 8))

# Inner Circle
ax.pie(inner_sizes, labels=None, radius=1, colors=inner_colors,
       startangle=72, counterclock=False, wedgeprops=dict(edgecolor='w'))
# Outer Circle
ax.pie(outer_sizes, labels=None, radius=1.3, colors=outer_colors,
       startangle=72, counterclock=False, wedgeprops=dict(width=0.3, edgecolor='w'))

plt.tight_layout()
# plt.savefig('analysis/assets/plots/distribution.pdf')
plt.show()
