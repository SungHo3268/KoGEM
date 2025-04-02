import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

dataset = json.load(open("datasets/KoGEM_benchmark.json", "r"))

prompts = []
prompt_lengths = []  # To store lengths of the prompts for analysis
lengths_passage_paragraph = []  # Both passage and paragraph exist
lengths_passage_only = []  # Only passage exists
lengths_paragraph_only = []  # Only paragraph exists
lengths_none = []  # Neither passage nor paragraph exist

for n, data in tqdm(enumerate(dataset), total=len(dataset),
                    bar_format="{l_bar}{bar:15}{r_bar}"):
    if data['data_src'] in ['NUAT(HS1)', 'NUAT(HS2)', 'NUAT(HS3)', 'CSAT']:
        cand_num = 5
    elif data['data_src'] in ['HSQE', 'LCSE(G9)', 'LCSE(G7)', 'NCSE(G9)', 'NCSE(G7)']:
        cand_num = 4
    else:
        raise NotImplementedError

    question = data['question'].strip()
    passage = data['passage'].strip()
    paragraph = data['paragraph'].strip()
    candidates = [cand.strip() for cand in data['candidates']]
    candidates = "\n ".join(candidates)

    # prompt
    if passage == '':
        if paragraph == '':
            prompt = f"질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
        else:
            prompt = f"설명: {paragraph} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
    else:
        if paragraph == '':
            prompt = f"지문: {passage} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "
        else:
            prompt = f"지문: {passage} 설명: {paragraph} 질문: 다음 선택지 1 부터 {cand_num} 중 {question}\n 선택지: {candidates}\n 정답: "

    prompts.append(prompt)
    label = str(data['label'])
    prompt_len = len(prompt)
    prompt_lengths.append(prompt_len)  # Collect the prompt length

    # Categorize the lengths based on passage and paragraph presence
    if passage and paragraph:
        lengths_passage_paragraph.append(prompt_len)
    elif passage:
        lengths_passage_only.append(prompt_len)
    elif paragraph:
        lengths_paragraph_only.append(prompt_len)
    else:
        lengths_none.append(prompt_len)

print(f"Total (#data): {np.mean([len(prompt) for prompt in prompts]):.2f} ({len(prompts)})")
print(f"Question + Choices (#data): {np.mean(lengths_none):.2f} ({len(lengths_none)})")
print(f"Question + Paragraph + Choices (#data): {np.mean(lengths_paragraph_only):.2f} ({len(lengths_paragraph_only)})")
print(f"Passage + Question + Choices (#data): {np.mean(lengths_passage_only):.2f} ({len(lengths_passage_only)})")
print(f"Passage + Question + Paragraph + Choices (#data): {np.mean(lengths_passage_paragraph):.2f} ({len(lengths_passage_paragraph)})")


# Plot the statistics of prompt lengths
plt.figure(figsize=(12, 8))

# Plot overlapping histograms for each group
plt.hist(lengths_passage_paragraph, bins=20, alpha=0.5, label='Passage + Question + Paragraph + Choices', color='blue', edgecolor='black')
plt.hist(lengths_passage_only, bins=20, alpha=0.5, label='Passage + Question + Choices', color='green', edgecolor='black')
plt.hist(lengths_paragraph_only, bins=20, alpha=0.5, label='Question + Paragraph + Choices', color='orange', edgecolor='black')
plt.hist(lengths_none, bins=20, alpha=0.5, label='Question + Choices', color='red', edgecolor='black')

# Add plot labels and legend
plt.title('Statistics of Prompt Lengths', fontsize=24)
plt.xlabel('Prompt Length', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig(f"analysis/assets/figures/prompt_length_statistics.pdf")
plt.savefig(f"analysis/assets/figures/prompt_length_statistics.png")
plt.show()

