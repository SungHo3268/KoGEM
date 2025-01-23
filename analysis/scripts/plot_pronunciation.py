"""
This file yields the Figure 5 in KoGEM paper
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 설정
data = {
    "Model": ["GPT-4o", "Claude-3.5-Sonnet", "o1-preview"],
    "Before": [44.60, 47.42, 71.83],
    "After": [47.89, 56.34, 74.65],
}

df = pd.DataFrame(data)
df["Improvement (%)"] = ((df["After"] - df["Before"]) / df["Before"]) * 100


sns.set_theme(style="whitegrid")  # Use seaborn pastel theme
plt.figure(figsize=(10, 6))
x = range(len(df["Model"]))
plt.bar(x, df["Before"], width=0.4, label="w/o pronunciation text", align='center', color='salmon')
plt.bar([i + 0.4 for i in x], df["After"], width=0.4, label="w/ pronunciation text", align='center', color='skyblue')
plt.xticks([i + 0.2 for i in x], df["Model"], fontsize=18)
plt.yticks(fontsize=14)
plt.xlabel(None)
plt.ylabel("Accuracy (%)", fontsize=18)
plt.ylim(30, 80)
plt.grid(True, linestyle='--', color='gray')  # Set grid with gray dotted lines
# plt.title("Performance Before and After Adding Pronunciation Information")
plt.legend(fontsize=14)
# Annotate improvement scores
for i, value in enumerate(df["Improvement (%)"]):
    plt.text(i + 0.2, df["After"][i] + 1, f"▲ {value:.1f}%", ha='center', fontsize=18, fontweight='bold',
             color='black')

plt.tight_layout()
# plt.savefig("analysis/assets/figures/add_pronunciation.pdf")
plt.show()
