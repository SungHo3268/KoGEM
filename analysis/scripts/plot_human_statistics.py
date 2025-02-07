"""
This file yields the Figure 6 in Appendix section of KoGEM paper
"""
import matplotlib.pyplot as plt


# Data for histograms (originally participants)
hsqe_accuracy = {"10s": 74.88, "20s": 79.23, "30s": 0, "40s": 0, "50s": 0, "60s": 0}
cse_accuracy = {"10s": 0, "20s": 49.13, "30s": 53.86, "40s": 50.34, "50s": 45.00, "60s": 43.33}

# Data for line graphs (originally participants)
hsqe = {"10s": 83, "20s": 26, "30s": 0, "40s": 0, "50s": 0, "60s": 0}
cse = {"10s": 0, "20s": 23, "30s": 92, "40s": 73, "50s": 52, "60s": 3}

# Process data for histograms
categories_hsqe = list(hsqe_accuracy.keys())
values_hsqe = list(hsqe_accuracy.values())

categories_cse = list(cse_accuracy.keys())
values_cse = list(cse_accuracy.values())

# Assign numeric labels to categorize for plotting
x_hsqe = [int(cat[:-1]) for cat in categories_hsqe]
x_cse = [int(cat[:-1]) for cat in categories_cse]

# Offset to separate the bars for clarity
offset = 2
x_hsqe_offset = [x - offset for x in x_hsqe]
x_cse_offset = [x + offset for x in x_cse]

# Process data for line graph
x_hsqe_acc = [x - offset for x in x_hsqe]
y_hsqe_acc = list(hsqe.values())

x_cse_acc = [x + offset for x in x_cse]
y_cse_acc = list(cse.values())

# Interpolation for smoothing
from scipy.interpolate import make_interp_spline
import numpy as np

def make_positive_from_here(y, left_to_right=True):
    if not left_to_right:
        y = y[::-1]
    for i in range(len(y)):
        if y[i] > 0:
            continue
        else:
            for j in range(i, len(y)):
                y[j] = 0
            break
    if not left_to_right:
        y = y[::-1]
    return y

x_hsqe_smooth = np.linspace(min(x_hsqe_acc), max(x_hsqe_acc), 500)
y_hsqe_smooth = make_interp_spline(x_hsqe_acc, y_hsqe_acc)(x_hsqe_smooth)
y_hsqe_smooth = make_positive_from_here(y_hsqe_smooth, left_to_right=True)

x_cse_smooth = np.linspace(min(x_cse_acc), max(x_cse_acc), 500)
y_cse_smooth = make_interp_spline(x_cse_acc, y_cse_acc)(x_cse_smooth)
y_cse_smooth = make_positive_from_here(y_cse_smooth, left_to_right=False)

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Histogram for HSQE (Accuracy)
ax1.bar(x_hsqe_offset, values_hsqe, width=4, alpha=0.7, label="HSQE Accuracy", color="skyblue", align="center")
# Histogram for CSE (Accuracy)
ax1.bar(x_cse_offset, values_cse, width=4, alpha=0.7, label="CSE Accuracy", color="orange", align="center")

# Set labels for the primary y-axis
ax1.set_xlabel("Age Groups", fontsize=18)
ax1.set_ylabel("Accuracy (%)", fontsize=18)
# ax1.set_title("HSQE and CSE with Participants Overlay")



# Secondary y-axis for participants
ax2 = ax1.twinx()
ax2.plot(x_hsqe_smooth, y_hsqe_smooth, color="blue", linewidth=2, label="HSQE Participants")
# ax2.scatter(x_hsqe_acc, y_hsqe_acc, color="blue", zorder=5, label="HSQE Data Points")
ax2.fill_between(x_hsqe_smooth, 0, y_hsqe_smooth, color="blue", alpha=0.2)
ax2.plot(x_cse_smooth, y_cse_smooth, color="red", linewidth=2, label="CSE Participants")
# ax2.scatter(x_cse_acc, y_cse_acc, color="red", zorder=5, label="CSE Data Points")
ax2.fill_between(x_cse_smooth, 0, y_cse_smooth, color="red", alpha=0.2)
ax2.set_ylabel("Number of Participants", fontsize=18)
ax2.spines["right"].set_position(("axes", 1.0))
ax2.set_ylim(bottom=0)

# Combine legends
lines_labels = ax1.get_legend_handles_labels()
lines_labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_labels[0] + lines_labels_2[0], lines_labels[1] + lines_labels_2[1],
           loc="upper right", fontsize=14)

# Set x-axis ticks
ax1.set_xticks(sorted(set(x_hsqe + x_cse)))
ax1.set_xticklabels([f"{x}s" for x in sorted(set(x_hsqe + x_cse))], fontsize=14)

ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)

# Display plot
plt.tight_layout()
plt.savefig("analysis/assets/figures/human_statistics_v2.pdf")
plt.show()

