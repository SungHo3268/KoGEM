import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


dataset = json.load(open("datasets/KoGEM_benchmark.json", "r"))
gold_labels = [data["label"] for data in dataset]
gold_labels = Counter(gold_labels)
gold_labels = {str(key): gold_labels[key] for key in [1, 2, 3, 4, 5]}

# eeve_outputs = json.load(open("logs/EEVE-Korean-Instruct-10.8B-v1.0/0_shot_predictions.json", "r"))
# eeve_predictions = [data["prediction"] for data in eeve_outputs]
# eeve_predictions = Counter(eeve_predictions)
# eeve_predictions = {key: eeve_predictions[key] for key in ['1', '2', '3', '4', '5']}

exaone_3_5_outputs = json.load(open("logs/EXAONE-3.5-32B-Instruct/0_shot_predictions.json", "r"))
exaone_3_5_predictions = [data["prediction"] for data in exaone_3_5_outputs]
exaone_3_5_predictions = Counter(exaone_3_5_predictions)
exaone_3_5_predictions = {key: exaone_3_5_predictions[key] for key in ['1', '2', '3', '4', '5']}

clovax_hcx_003_outputs = json.load(open("logs/ClovaX/HCX-003_0_shot_predictions.json", "r"))
clovax_hcx_003_predictions = [data["prediction"] for data in clovax_hcx_003_outputs]
clovax_hcx_003_predictions = Counter(clovax_hcx_003_predictions)
clovax_hcx_003_predictions = {key: clovax_hcx_003_predictions[key] for key in ['1', '2', '3', '4', '5']}

qwen_2_5_outputs = json.load(open("logs/Qwen2.5-32B-Instruct/0_shot_predictions.json", "r"))
qwen_2_5_predictions = [data["prediction"] for data in qwen_2_5_outputs]
qwen_2_5_predictions = Counter(qwen_2_5_predictions)
qwen_2_5_predictions = {key: qwen_2_5_predictions[key] for key in ['1', '2', '3', '4', '5']}

claude_3_5_sonnet_outputs = json.load(open("logs/Claude/claude-3-5-sonnet-20240620_0_shot_predictions.json", "r"))
claude_3_5_sonnet_predictions = [data["prediction"] for data in claude_3_5_sonnet_outputs]
claude_3_5_sonnet_predictions = Counter(claude_3_5_sonnet_predictions)
claude_3_5_sonnet_predictions = {key: claude_3_5_sonnet_predictions[key] for key in ['1', '2', '3', '4', '5']}

# gpt_3_5_outputs = json.load(open("logs/OpenAI/gpt-3.5-turbo-0125_0_shot_predictions.json", "r"))
# gpt_3_5_predictions = [data["prediction"] for data in gpt_3_5_outputs]
# gpt_3_5_predictions = Counter(gpt_3_5_predictions)
# gpt_3_5_predictions = {key: gpt_3_5_predictions[key] for key in ['1', '2', '3', '4', '5']}

gpt_4o_outputs = json.load(open("logs/OpenAI/gpt-4o_0_shot_predictions.json", "r"))
gpt_4o_predictions = [data["prediction"] for data in gpt_4o_outputs]
gpt_4o_predictions = Counter(gpt_4o_predictions)
gpt_4o_predictions = {key: gpt_4o_predictions[key] for key in ['1', '2', '3', '4', '5']}

o1_preview_outputs = json.load(open("logs/OpenAI/o1-preview_0_shot_predictions.json", "r"))
o1_preview_predictions = [data["prediction"] for data in o1_preview_outputs]
o1_preview_predictions = Counter(o1_preview_predictions)
o1_preview_predictions = {key: o1_preview_predictions[key] for key in ['1', '2', '3', '4', '5']}

# Consolidate all predictions into a dictionary and normalize to percentages
all_predictions = {
    source: {label: (count / sum(predictions.values())) * 100 for label, count in predictions.items()}
    for source, predictions in {
        "Gold Labels": gold_labels,
        # "EEVE": eeve_predictions,
        "EXAONE-3.5-32B": exaone_3_5_predictions,
        "ClovaX-HCX-003": clovax_hcx_003_predictions,
        "Qwen-2.5-32B": qwen_2_5_predictions,
        "Claude-3.5-Sonnet": claude_3_5_sonnet_predictions,
        # "GPT-3.5-Turbo": gpt_3_5_predictions,
        "GPT-4o": gpt_4o_predictions,
        "o1-preview": o1_preview_predictions
    }.items()
}

# Define labels and percentages for the x-axis
x_labels = ['1', '2', '3', '4', '5']
data_sources = list(all_predictions.keys())
counts = np.array([[all_predictions[source].get(label, 0) for label in x_labels] for source in data_sources])

# Plotting
width = 0.1  # Width of individual bars
x_pos = np.arange(len(x_labels))

colors = ['#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#76b7b2', '#edc948', '#b07aa2']  # Harmonious color palette
plt.figure(figsize=(12, 8))
for i, source in enumerate(data_sources):
    plt.bar(x_pos + i * width, counts[i], width=width, label=source, color=colors[i % len(colors)])

# Customization
plt.xticks(x_pos + width * (len(data_sources) / 2 - 0.5), x_labels, fontsize=16)  # Center x-ticks
plt.yticks(fontsize=16)
plt.xlabel("Predictions", fontsize=18)
plt.ylabel("Percentage (%)", fontsize=18)
plt.title("Label Distributions Across Models", fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig("analysis/assets/figures/label_distributions.pdf")
plt.savefig("analysis/assets/figures/label_distributions.png")
plt.show()

print(f"[ Class Distributions as Percentages ]")
for source, predictions in all_predictions.items():
    trimmed_dict = {key: round(predictions[key], 1) for key in predictions}
    print(f"{source}: {trimmed_dict}")
