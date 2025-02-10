"""
This file yields the Figure 5 in KoGEM paper
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# # 데이터 설정
# data = {
#     "Model": ["GPT-4o", "Claude-3.5-Sonnet", "o1-preview"],
#     "Before": [34.48, 48.28, 72.41],
#     "After": [41.38, 51.72, 86.21],
# }

data = {
    "Model": ["Claude-3-Haiku", "GPT-4o", "Claude-3.5-Sonnet", "o1-preview"],
    "Before": [31.03, 34.48, 48.28, 72.41],
    "After": [31.03, 41.38, 51.72, 86.21],
}



df = pd.DataFrame(data)
df["Improvement (%)"] = ((df["After"] - df["Before"]) / df["Before"]) * 100


sns.set_theme(style="whitegrid")  # Use seaborn pastel theme
sns.set_theme(style="whitegrid")  # Use seaborn pastel theme
plt.figure(figsize=(10, 6))
x = range(len(df["Model"]))
plt.bar(x, df["Before"], width=0.4, label="w/o morpheme text", align='center', color='salmon')
plt.bar([i + 0.4 for i in x], df["After"], width=0.4, label="w/ morpheme text", align='center', color='lightgreen')
plt.xticks([i + 0.2 for i in x], df["Model"], fontsize=18)
plt.yticks(fontsize=16)
plt.xlabel(None)
plt.ylabel("Accuracy (%)", fontsize=20)
plt.ylim(0, 100)
plt.grid(False, linestyle='--', color='white')  # Set grid with gray dotted lines
# plt.title("Performance Before and After Adding Pronunciation Information")
plt.legend(fontsize=18, loc='upper left')
# Annotate improvement scores
for i, value in enumerate(df["Improvement (%)"]):
    plt.text(i + 0.2, df["After"][i] + 1, f"▲ {value:.1f}%", ha='center', fontsize=20, fontweight='bold', color='black')

plt.tight_layout()
plt.savefig("analysis/assets/figures/add_morpheme_v2.pdf")
plt.show()
