import json

"""
[ API CALL ]
OpenAI
: gpt-3.5-turbo-0125  ||  gpt-4o  ||  gpt-4o-mini  ||  o1-preview

Claude
: claude-3-5-sonnet-20240620  ||  claude-3-haiku-20240307

Gemini
: gemini-2.0-flash-exp || gemini-1.5-flash

Llama
: llama3-70b  ||  llama3.1-405b

ClovaX
: HCX-DASH-001  ||  HCX-003

[ HF CALL ]
llama-3-Korean-Bllossom-8B
SOLAR-10.7B-Instruct-v1.0
KULLM3
EEVE-Korean-10.8B-v1.0
EEVE-Korean-Instruct-10.8B-v1.0
EXAONE-3.0-7.8B-Instruct
EXAONE-3.5-7.8B-Instruct
EXAONE-3.5-32B-Instruct
gemma-2-9b-it
gemma-2-27b-it
Qwen2.5-7B-Instruct
Qwen2.5-14B-Instruct
Qwen2.5-32B-Instruct
DeepSeek-R1-Distill-Qwen-14B
DeepSeek-R1-Distill-Qwen-32B
s1-32B
Llama-3.1-8B-Instruct
"""

model_type = "s1-32B"
model_var = "gpt-4o-mini"
eval_method = "0"          # 0, 1, 5, 'each_major_one', 'each_sub_one'
cot = False
cot_file_index = "_cot" if cot else ""


if model_type in ["OpenAI", "Claude", "Gemini", "Llama", "ClovaX"]:
    total_dataset = json.load(open(f"logs/{model_type}/{model_var}_{eval_method}_shot{cot_file_index}_predictions.json", "r"))
else:
    total_dataset = json.load(open(f"logs/{model_type}/{eval_method}_shot{cot_file_index}_predictions.json", "r"))


kogem_info = json.load(open("utils/KoGEM_info.json", "r"))
major = kogem_info["major_categories"]
sub = kogem_info["sub_categories"]
data_srcs = kogem_info["data_srcs"]


datasets_per_srcs = {src: [] for src in data_srcs}
for data in total_dataset:
    src = data['data_src']
    if src not in data_srcs:
        raise NotImplementedError

    datasets_per_srcs[src].append(data)


results_per_srcs = []
for data_src in datasets_per_srcs:
    dataset = datasets_per_srcs[data_src]

    major_results = {cat: [0, 0] for cat in major}
    sub_results = {cat: [0, 0] for cat in sub}
    for data in dataset:
        label = int(data['label'])
        try:
            prediction = int(data['prediction'])
        except ValueError:
            prediction = -1
        level_1 = data['level_1']
        level_2 = data['level_2']

        cor = (label == prediction) * 1
        major_results[level_1][0] += cor
        major_results[level_1][1] += 1

        sub_results[level_2][0] += cor
        sub_results[level_2][1] += 1

    print("############################################")
    print(f"[ Data Source: {data_src} ]")
    print("############################################")


    print("Total Results")
    cor = sum([major_results[cat][0] for cat in major_results])
    total = sum([major_results[cat][1] for cat in major_results])
    print(f"{cor}/{total} ({cor / total * 100:.2f} [%])")
    print("")
    results_per_srcs.append(f"{cor / total * 100:.2f}")

    print("Major Categories")
    for cat in major_results:
        cor, total = major_results[cat]
        if total == 0:
            # print(f"{cat}: -")
            print(f"-")
        else:
            # print(f"{cat}: {cor}/{total} ({cor/total*100:.2f} [%])")
            print(f"{cor / total * 100:.2f}")
    print("")

    print("Sub Categories")
    for cat in sub_results:
        cor, total = sub_results[cat]
        if total == 0:
            # print(f"{cat}: -")
            print(f"-", end="\t")
        else:
            # print(f"{cat}: {cor}/{total} ({cor/total*100:.2f} [%])")
            print(f"{cor / total * 100:.2f}", end="\t")
    print("\n")


results_per_srcs = "\t".join(results_per_srcs)
print(f"For Excel Copy-Paste")
print(results_per_srcs)
